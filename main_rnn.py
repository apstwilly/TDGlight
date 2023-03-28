import os
import json
import argparse
from datetime import datetime

import cityflow
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.DQN_rnn import DeepQNetwork

total_reward = []
eval_reward = []
test_reward = []
throughput = []
best_reward = [10000000, 10000000]
action_list = []
total_action_list = []


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--reward_decay", type=float, default=0.8)
    parser.add_argument("--ob_decay", type=float, default=0.6)
    parser.add_argument("--e_greedy", type=float, default=0.99)
    parser.add_argument("--e_greedy_increment", type=float, default=0.0005)
    parser.add_argument("--replace_target_iter", type=int, default=300)
    parser.add_argument("--memory_size", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--load_weight", type=str, default=None)
    parser.add_argument("--n_actions", type=int, default=8)
    parser.add_argument("--n_features", type=int, default=16)
    parser.add_argument("--n_gcn_features", type=int, default=2)

    parser.add_argument("--action_period", type=int, default=20)
    parser.add_argument("--yellow_time", type=int, default=5)
    parser.add_argument("--map_size", type=str, default="1X1")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--train_epoch_len", type=int, default=3600)
    parser.add_argument("--test_epoch_len", type=int, default=3600)
    parser.add_argument("--output_graph", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--action", action="store_true")

    return parser.parse_args()


class Interection:
    def __init__(self, inter_id, args):
        self.inter_id = inter_id
        self.inter_name = "intersection_{0}_{1}".format(inter_id[0], inter_id[1])
        self.args = args
        self.lane_in = self.get_in()
        self.lane_out = self.get_out()
        self.lane = self.get_total()
        self.p_ph = -1
        self.dur = 0
        self.action = 0

    def get_road_name(self, inter, direction=1):
        output = []
        output.append("road_" + inter + "_0")
        output.append("road_" + inter + "_1")
        if direction == 1:
            return list(output)
        output.reverse()
        return list(output)

    def get_in(self):
        inter1 = self.inter_id[0]
        inter2 = self.inter_id[1]

        output = []
        output.extend(self.get_road_name("{}_{}_0".format(inter1 - 1, inter2), 1))
        output.extend(self.get_road_name("{}_{}_1".format(inter1, inter2 - 1), 1))
        output.extend(self.get_road_name("{}_{}_2".format(inter1 + 1, inter2), 1))
        output.extend(self.get_road_name("{}_{}_3".format(inter1, inter2 + 1), 1))
        return output

    def get_out(self):
        inter1 = self.inter_id[0]
        inter2 = self.inter_id[1]

        output = []
        output.extend(self.get_road_name("{}_{}_2".format(inter1, inter2), 0))
        output.extend(self.get_road_name("{}_{}_3".format(inter1, inter2), 0))
        output.extend(self.get_road_name("{}_{}_0".format(inter1, inter2), 0))
        output.extend(self.get_road_name("{}_{}_1".format(inter1, inter2), 0))
        return output

    def get_total(self):
        inter1 = self.inter_id[0]
        inter2 = self.inter_id[1]

        output = []
        output.extend(self.get_road_name("{}_{}_2".format(inter1, inter2), 0))
        output.extend(self.get_road_name("{}_{}_0".format(inter1 - 1, inter2), 1))
        output.extend(self.get_road_name("{}_{}_3".format(inter1, inter2), 0))
        output.extend(self.get_road_name("{}_{}_1".format(inter1, inter2 - 1), 1))
        output.extend(self.get_road_name("{}_{}_0".format(inter1, inter2), 0))
        output.extend(self.get_road_name("{}_{}_2".format(inter1 + 1, inter2), 1))
        output.extend(self.get_road_name("{}_{}_1".format(inter1, inter2), 0))
        output.extend(self.get_road_name("{}_{}_3".format(inter1, inter2 + 1), 1))
        return output

    def get_observation(self, lane_car, lane_wait):
        observation = []
        for i in self.lane:
            tem_num = lane_car[i]
            tem_wait = lane_wait[i]
            observation.append(tem_num)
            observation.append(tem_wait)
        return np.array(observation)

    def get_reward(self, wait_count):
        tem_reward = 0
        for i in self.lane_in:
            tem_reward = tem_reward + wait_count[i]
        return tem_reward

    def get_light(self, n_ph):
        if self.action == -1:
            if self.dur == self.args.yellow_time:
                self.action = n_ph
                self.dur = 0
        elif self.p_ph != n_ph:
            self.action = -1
            self.dur = 0
        elif self.p_ph == n_ph:
            self.action = n_ph
            self.dur = 0

        self.dur = self.dur + 1
        self.p_ph = n_ph
        return self.action


def write_reslut(path, name, data, graph=True, ind=True):
    file = "{}/{}.txt".format(path, name)
    log_fd = open(file, "w")
    for i, v in enumerate(data):
        if ind:
            log_fd.write("{} : {}\n".format(i, v))
        else:
            for j in v:
                log_fd.write("{} ".format(j))
            log_fd.write("\n")
    log_fd.close()

    if graph:
        plt.figure()
        plt.plot(np.arange(len(data)), data)
        plt.ylabel("Reward")
        plt.xlabel("training steps")
        file = "{}/{}.png".format(path, name)
        plt.savefig(file)


def write_args(path, args):
    f = open(model_path + "/config.txt", "w")
    args_dic = vars(args)
    args_key = args_dic.keys()
    for k in args_key:
        f.write("{} : {}\n".format(k, str(args_dic[k])))
    f.close()


def write_config(path, map_size, replay=False):
    cityflow_config = {
        "interval": 1.0,
        "seed": 0,
        "dir": "./data/" + map_size + "/",
        "roadnetFile": "roadnet.json",
        "flowFile": "flow.json",
        "rlTrafficLight": True,
        "laneChange": False,
        "saveReplay": replay,
        "roadnetLogFile": "../../" + path + "/replay_roadnet.json",
        "replayLogFile": "../../" + path + "/replay.txt",
    }

    if replay:
        with open(path + "/cityflow_replay.json", "w") as json_file:
            json.dump(cityflow_config, json_file)
    else:
        with open(path + "/cityflow.json", "w") as json_file:
            json.dump(cityflow_config, json_file)


def _train(RL, env, total_int, intersection, episode, total_epoch, args):
    if episode >= 2:
        print("training...........")
        for i in tqdm(range(200)):
            RL.learn()

    env.reset(True)

    his = []
    ob_buf = [None] * total_int
    re_buf = np.zeros([total_int])
    observation = [[] for _ in range(total_int)]
    env_lane_num = env.get_lane_vehicle_count()
    env_lane_wait = env.get_lane_waiting_vehicle_count()
    for i in range(total_int):
        observation[i].append(
            intersection[i].get_observation(env_lane_num, env_lane_wait)
        )
    his.append(observation)

    # t_reward = 0
    epoch_len = args.train_epoch_len
    print("generating...........")
    for step in tqdm(range(epoch_len)):
        if step % args.action_period == 0:
            n_ph = []
            for i in range(total_int):
                n_ph.append(RL.choose_action(np.stack(observation[i], axis=0)))
            action_list.append(n_ph)

        light = []
        for i in range(total_int):
            light.append(intersection[i].get_light(n_ph[i]) + 1)
            env.set_tl_phase(intersection[i].inter_name, light[i])
        total_action_list.append(light)
        env.next_step()

        env_lane_num = env.get_lane_vehicle_count()
        env_lane_wait = env.get_lane_waiting_vehicle_count()

        observation_ = []
        for i in range(total_int):
            observation_.append(
                observation[i]
                + intersection[i]
                .get_observation(env_lane_num, env_lane_wait)
                .reshape(1, -1)
                .tolist()
            )

        if step % args.action_period == 0:
            for i in range(total_int):
                ob_buf[i] = observation[i]
                re_buf[i] = intersection[i].get_reward(env_lane_wait)
        elif step % args.action_period == args.action_period - 1:
            for i in range(total_int):
                tem_r = intersection[i].get_reward(env_lane_wait)
                RL.store_transition(
                    ob_buf[i], n_ph[i], re_buf[i] - tem_r, observation_[i]
                )

        # Update observation
        for i in range(total_int):
            observation[i] = observation_[i]
            print(f"Inter{i}:", len(observation[i]))

    # total_reward.append(t_reward)
    eval_reward.append(float(env.get_average_travel_time()))
    print("Reward_travel : {}".format(float(env.get_average_travel_time())))

    return RL


def _test(RL, env, total_int, intersection, args):
    env.reset(True)

    epoch_len = args.test_epoch_len
    observation = [[] for _ in range(total_int)]
    env_lane_num = env.get_lane_vehicle_count()
    env_lane_wait = env.get_lane_waiting_vehicle_count()
    for i in range(total_int):
        observation[i].append(
            intersection[i].get_observation(env_lane_num, env_lane_wait)
        )

    epoch_len = args.test_epoch_len
    car_in = []
    print("testing...........")
    for step in tqdm(range(epoch_len)):
        if step % 20 == 0 and args.throughput:
            for c in env.get_vehicles(include_waiting=False):
                if c not in car_in:
                    car_in.append(c)
        if step % args.action_period == 0:
            n_ph = []
            for i in range(total_int):
                n_ph.append(RL.choose_action(np.stack(observation[i], axis=0)))
        for i in range(total_int):
            env.set_tl_phase(
                intersection[i].inter_name, intersection[i].get_light(n_ph[i]) + 1
            )
        env.next_step()
        env_lane_num = env.get_lane_vehicle_count()
        env_lane_wait = env.get_lane_waiting_vehicle_count()

        for i in range(total_int):
            observation[i].append(
                intersection[i].get_observation(env_lane_num, env_lane_wait)
            )

    if args.throughput:
        cars = list(env.get_vehicles(include_waiting=False))
        for c in cars:
            if c not in car_in:
                car_in.append(c)
        num_car_in = len(car_in)
        num_car_remain = len(cars)
        for c in env.get_vehicles(include_waiting=True):
            if c not in car_in:
                car_in.append(c)
        num_car_total = len(car_in)
    else:
        num_car_total = 0
        num_car_in = 0
        num_car_remain = 0

    test_reward.append(float(env.get_average_travel_time()))
    throughput.append([num_car_total, num_car_in, num_car_in - num_car_remain])
    print(
        "test_travel : {},  total : {},  in : {},  out : {}".format(
            float(env.get_average_travel_time()),
            num_car_total,
            num_car_in,
            num_car_in - num_car_remain,
        )
    )


def run(args, model_path):
    write_config(model_path, args.map_size)
    env = cityflow.Engine(model_path + "/cityflow.json", thread_num=1)
    total_epoch = args.epoch
    map_size = args.map_size.split("_")[0]
    hight = int(map_size.split("X")[0])
    width = int(map_size.split("X")[1])
    total_int = hight * width

    intersection = []
    for i in range(1, hight + 1):
        for j in range(1, width + 1):
            intersection.append(Interection([j, i], args))

    RL = DeepQNetwork(args)

    print("************env name : ", map_size)
    print("".join(f"{k} : {v}\n" for k, v in vars(args).items()))

    save_path = os.path.join(model_path, args.map_size)

    for episode in range(total_epoch):
        print("\n\n")
        print("Epoch : {}/{}".format(episode + 1, total_epoch))
        RL = _train(RL, env, total_int, intersection, episode, total_epoch, args)

        if episode >= int(total_epoch / 2):
            if eval_reward[episode] < best_reward[1]:
                RL.save_model(save_path + "/params")
                # best_reward[0] = total_reward[episode]
                best_reward[1] = eval_reward[episode]

    if args.replay:
        write_config(model_path, args.map_size, args.replay)
        env = cityflow.Engine(model_path + "/cityflow_replay.json", thread_num=1)
        _test(RL, env, total_int, intersection, args)
    elif args.throughput:
        write_config(model_path, args.map_size)
        env = cityflow.Engine(model_path + "/cityflow.json", thread_num=1)
        _test(RL, env, total_int, intersection, args)

    # return RL.cost_his


if __name__ == "__main__":
    args = parse_args()
    name = args.map_size
    time = datetime.now().strftime("%m%d_%H%M")
    model_path = "model"
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_path = os.path.join("model/", name + "_" + time)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    write_args(model_path, args)

    try:
        run(args, model_path)
    except KeyboardInterrupt:
        print("interrupt")
    finally:
        print("game over")
        # write_reslut(path, 'reward', total_reward)
        write_reslut(model_path, "reward_aver", eval_reward)
        if args.action:
            write_reslut(model_path, "action", action_list, graph=False, ind=False)
            write_reslut(
                model_path, "total_action", total_action_list, graph=False, ind=False
            )
        if args.replay or args.throughput:
            write_reslut(model_path, "reward_test", test_reward)
            write_reslut(model_path, "throughput", throughput, graph=False)
