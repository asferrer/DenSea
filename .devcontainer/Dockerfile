FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Establece variables de entorno
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Instala las dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Crea un enlace simbólico para python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Actualiza pip
RUN pip install --upgrade pip

# Instala PyTorch con CUDA 12.1 (ajusta las versiones si es necesario)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Instala Detectron2
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Instala timm
RUN pip install timm

# Establece el directorio de trabajo
WORKDIR /app/DiffusionDet

# CMD default para que VS Code pueda interactuar
CMD ["sleep", "infinity"]