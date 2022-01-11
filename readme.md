## Installation

We recommend to run the code through docker.

1. Please clone this code.
2. Please pull the docker image of cityflow, and build the container.

```
docker pull cityflowproject/cityflow:latest
docker run -it --name city -v path/to/the/code/repo/:/home/myfile cityflowproject/cityflow:latest
```

More brief documentation can be found at :
https://docs.docker.com/
https://readthedocs.org/projects/cityflow/downloads/pdf/latest/

## Usage

Running the default setting.

```
python main.py
```

Or running model have already trained well.

```
python main.py --map_size=3X4 --load_weight=load/3X4
```


