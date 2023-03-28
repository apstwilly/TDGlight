# TeDA-GCRL

## Installation

We recommend to run the code through docker.

1. Please clone this repo.

    ```shell
    git clone https://github.com/apstwilly/TDGlight/tree/main
    ```

2. Please pull the docker image of cityflow, and build the container.

    ```shell
    docker pull cityflowproject/cityflow:latest
    docker run -it --name city -v path/to/the/code/repo/:/home/myfile cityflowproject/cityflow:latest
    ```

More brief documentation can be found at :

1. [Docker Documentation](https://docs.docker.com/)
2. [CityFlow Documentation](https://readthedocs.org/projects/cityflow/downloads/pdf/latest/)

> To run the code in a GPU environment, please refer to our [here](docker/README.md).

## Usage

Running the default setting.

```shell
python main.py
```

Or running model have already trained well.

```shell
python main.py --map_size=3X4 --load_weight=load/3X4
```

> To use `RNN` version, replace `main.py` with `main_rnn.py`