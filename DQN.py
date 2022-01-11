import numpy as np
import tensorflow as tf
import scipy.sparse as sp

np.random.seed(1)
tf.set_random_seed(777)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_features = args.n_features
        self.n_gcn_features = args.n_gcn_features
        self.lr = args.learning_rate
        self.gamma = args.reward_decay
        self.epsilon_max = args.e_greedy
        self.replace_target_iter = args.replace_target_iter
        self.memory_size = args.memory_size
        self.batch_size = args.batch_size
        self.epsilon_increment = args.e_greedy_increment
        self.epsilon = 0 if args.e_greedy_increment > 0 else self.epsilon_max
        
        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, self.n_features * self.n_gcn_features * 2 + 2))

        
        self.A_hat = self.get_adm('./ADM/traffic.txt')

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if args.output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logss/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
        self.saver = tf.train.Saver()
        if(args.load_weight):
            self.load_weight(args.load_weight)
                
    def get_adm(self, file, sym=False):
        
        struct_edges = np.genfromtxt(file, dtype=np.int32)
        sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
        sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 1], sedges[:, 0])), shape=(self.n_features, self.n_features), dtype=np.float32)
        if sym:
            sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
        matirx_A = np.array(sadj.toarray()).astype(np.float32)            
             
     
        I = np.matrix(np.eye(self.n_features)).astype(np.float32)
        A_hat = matirx_A + I
        
        A_hat = A_hat.astype(np.float32)   
        
        D = np.array(np.sum(A_hat, axis=1)).reshape(-1).astype(np.float32)         
        D = np.matrix(np.diag(D)).astype(np.float32)        
        A_hat = (D**-1*A_hat).astype(np.float32)
        return A_hat
    
    def GCN(self, name, adm, input_s):
        gcnf = self.n_gcn_features
        gcn_layer = [gcnf, gcnf, gcnf]
        gcnw1 = tf.get_variable(name = name + 'gcnw1', shape=(self.n_gcn_features, gcn_layer[0]), initializer=tf.initializers.glorot_uniform())        
        gcnw2 = tf.get_variable(name = name + 'gcnw2', shape=(gcn_layer[0], gcn_layer[1]), initializer=tf.initializers.glorot_uniform())            
        gcnw3 = tf.get_variable(name = name + 'gcnw3', shape=(gcn_layer[1], gcn_layer[2]), initializer=tf.initializers.glorot_uniform())
        
        ex = tf.matmul(adm, input_s)
        ex = tf.matmul(ex, gcnw1)
        ex = tf.matmul(adm, ex)
        ex = tf.matmul(ex, gcnw2)
        ex = tf.matmul(adm, ex)
        ex = tf.matmul(ex, gcnw3)       
        #ex = tf.transpose(ex)
        #ex = tf.nn.leaky_relu(ex)
        ex = tf.nn.leaky_relu(ex)
        
        return ex
        
    def get_net(self, input_s):
        
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        
        hidden_layer = [64, 32]
        
        
        ex = self.GCN('matirx_A_', self.A_hat, input_s)        
        #ex = input_s
        '''
        ex = tf.reshape(ex, [1, 1, self.n_features*self.n_gcn_features])
        gru_e = tf.keras.layers.LSTM(self.n_features*self.n_gcn_features, 
                                    recurrent_initializer='orthogonal',
                                    bias_initializer='ones',
                                    recurrent_regularizer = tf.keras.regularizers.l2(0.01),
                                    dropout=0.2, recurrent_dropout=0.2,)
        ex = gru_e(ex)
        '''
        e0 = tf.reshape(ex, [1, self.n_features*self.n_gcn_features])
        e1 = tf.layers.dense(e0, hidden_layer[0], tf.nn.leaky_relu, kernel_initializer=w_initializer,
                             bias_initializer=b_initializer, name='e1')
        e2 = tf.layers.dense(e1, hidden_layer[1], tf.nn.leaky_relu, kernel_initializer=w_initializer,
                             bias_initializer=b_initializer, name='e2')
        output = tf.layers.dense(e2, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')
        return output
        
    def _build_net(self):
        
        tf.reset_default_graph()
        
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [self.n_features, self.n_gcn_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [self.n_features, self.n_gcn_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None,], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None,], name='a')  # input Action

        with tf.variable_scope('eval_net'):
            
            self.q_eval = self.get_net(self.s)

        with tf.variable_scope('target_net'):
            
            self.q_next = self.get_net(self.s_)

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * np.max(self.q_next)
            self.q_target = tf.stop_gradient(q_target)
            
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
            #self.q_eval_wrt_a = self.q_eval_wrt_a*self.q_eval_wrt_a
            
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
            
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(self.lr, epsilon=1e-08)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            self._train_op = optimizer.apply_gradients(zip(gradients, variables))
            
            
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        observation = observation.reshape(self.n_features, self.n_gcn_features)

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            #print('\ntarget_params_replaced')
            self.sess.run(self.target_replace_op)

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        
        
        t_cost = 0
        
        for i in batch_memory:
            i = np.array(i)
            _, cost = self.sess.run(
                [self._train_op, self.loss],
                feed_dict={
                    self.s: i[:self.n_features*self.n_gcn_features].reshape(self.n_features, self.n_gcn_features),
                    self.a: [i[self.n_features*self.n_gcn_features]],
                    self.r: [i[self.n_features*self.n_gcn_features + 1]],
                    self.s_: i[self.n_features*self.n_gcn_features + 2:].reshape(self.n_features, self.n_gcn_features),
                })
            t_cost = t_cost + cost

        self.cost_his.append(t_cost)

        self.learn_step_counter += 1

    def load_weight(self, load_weight):
        checkpoint = tf.train.get_checkpoint_state(load_weight)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path) 
            self.epsilon = self.epsilon_max    
            print('----------load model------------')
        else:
            print('----------load model fail------------')

    def save_model(self, path):
        print('*************************model save')
        self.saver.save(self.sess, path)

    def set_replace(self, replace):
        self.replace_target_iter = replace
        print('reset the replace target iter : ',self.replace_target_iter) 
    
    def set_lr(self, lr):
        self.lr = lr
        print('reset the learning rate : ',self.lr) 
    