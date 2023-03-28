#####
# This Dockerfile support users to use GPU in CityFlow docker container. Follow below step to build the docker image.
# 1. Git clone https://github.com/cityflow-project/CityFlow
# 2. Replace their original Dockerfile with the one you are reading now.
# 3. Build the docker image with command: docker (podman) build -t cityflow-gpu:lastest .
# All done!
#####

# User should check their driver version
# See https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html
FROM nvcr.io/nvidia/tensorflow:22.01-tf1-py3

# c++ dependencies
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt update && \
    apt-get install -y build-essential cmake wget git

# install Miniconda Python 3.6
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN wget -P /tmp/ https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh && \
    /bin/bash /tmp/Miniconda3-4.5.4-Linux-x86_64.sh -b -p /opt/conda && \
    rm /tmp/Miniconda3-4.5.4-Linux-x86_64.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# install cityflow
COPY . /home/cityflow
RUN pip install flask && \
    cd /home/cityflow && \
    pip install . \
    pip install --upgrade pip
