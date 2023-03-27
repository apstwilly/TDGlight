# Using GPU with CityFlow

To enable GPU support, users should build their CityFlow container by their own.

> Note that users need to install the NVIDIA Container Toolkit. Please refer to the [official documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) for more details.

## Installation

1. Clone the repo from CityFlow.
2. Replace their Dockerfile with [ours](./Dockerfile) in this directory.
3. Use the [`requirements.txt`](./requirements.txt) in this folder to install the dependencies within the container.
