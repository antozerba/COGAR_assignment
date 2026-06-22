# Base image Isaac Sim 5.1.0
FROM nvcr.io/nvidia/isaac-sim:5.1.0

ENV DEBIAN_FRONTEND=noninteractive

#User root
USER root

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    curl \
    wget \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY IsaacLab /root/IsaacLab

WORKDIR /root/IsaacLab
RUN /isaac-sim/python.sh -m pip install --upgrade pip && \
    /isaac-sim/python.sh -m pip install -e source/isaaclab && \
    /isaac-sim/python.sh -m pip install -e source/isaaclab_assets && \
    /isaac-sim/python.sh -m pip install -e source/isaaclab_tasks

RUN /isaac-sim/python.sh -m pip install rsl-rl-lib && \ 
    ln -s /isaac-sim /root/IsaacLab/_isaac_sim  && \
    /isaac-sim/python.sh -m pip install -e source/isaaclab_rl 


WORKDIR /isaac-sim

ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV OMNI_KIT_ALLOW_ROOT=1

CMD ["/bin/bash"]