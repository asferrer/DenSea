FROM ubuntu:18.04
ARG DEBIAN_FRONTEND=noninteractive

ARG USERNAME=user
ARG USER_UID=$USER_UID
ARG USER_GID=$USER_UID

RUN apt-get update && \
    apt-get install -y python3.7 python3-pip ffmpeg libsm6 libxext6 && \
    python3.7 -m pip install --upgrade pip && \
    pip install --upgrade requests

RUN wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1604-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu1604-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb && \
    apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub && \
    apt-get update && \
    apt-get -y install cuda-10.2
    
# Set the working directory
WORKDIR /app

RUN python3.7 -m pip install --upgrade pip

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo requirements.txt al contenedor
COPY requirements.txt /app/requirements.txt

# Instalar dependencias, asegurándose de que pycocotools se compile correctamente
RUN python3.7 -m pip install Cython && \ 
    python3.7 -m pip install setuptools==58.2.0 detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html && \
    python3.7 -m pip install -r requirements.txt 

# Copiar el resto de los archivos del proyecto al contenedor
COPY . /app
