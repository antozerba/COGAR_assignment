# Base image Isaac Sim 5.1.0
FROM nvcr.io/nvidia/isaac-sim:5.1.0

# Evita prompt interattivi durante installazione
ENV DEBIAN_FRONTEND=noninteractive

#User root
USER root

# Installa dipendenze di sistema
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    curl \
    wget \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copia il repo IsaacLab nell'immagine
COPY IsaacLab /root/IsaacLab

# Installa Isaac Lab usando il Python interno di Isaac Sim
WORKDIR /root/IsaacLab
RUN /isaac-sim/python.sh -m pip install --upgrade pip && \
    /isaac-sim/python.sh -m pip install -e source/isaaclab && \
    /isaac-sim/python.sh -m pip install -e source/isaaclab_assets && \
    /isaac-sim/python.sh -m pip install -e source/isaaclab_tasks
    # /isaac-sim/python.sh -m pip install -e source/isaaclab_mirage 2>/dev/null || true && \
    # /isaac-sim/python.sh -m pip install -e source/isaaclab_rl 2>/dev/null || true 

RUN /isaac-sim/python.sh -m pip install rsl-rl-lib && \ 
    ln -s /isaac-sim /root/IsaacLab/_isaac_sim  && \
    /isaac-sim/python.sh -m pip install -e source/isaaclab_rl 


# Torna alla directory di Isaac Sim
WORKDIR /isaac-sim

# Variabili ambiente
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV OMNI_KIT_ALLOW_ROOT=1

CMD ["/bin/bash"]