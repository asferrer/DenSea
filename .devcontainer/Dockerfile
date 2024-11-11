FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Establecer variables de entorno para evitar interacciones durante la instalación
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Instalar dependencias del sistema y limpiar para reducir el tamaño de la imagen
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        python3 \
        python3-pip \
        python3-dev \
        libgl1 \
        libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Crear un enlace simbólico para 'python'
RUN ln -s /usr/bin/python3 /usr/bin/python

# Actualizar 'pip' y instalar paquetes de Python necesarios en una sola capa
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir \
        'git+https://github.com/facebookresearch/detectron2.git' && \
    pip install --no-cache-dir \
        timm \
        opencv-python \
        seaborn \
        scikit-learn \
        streamlit

# Establecer el directorio de trabajo
WORKDIR /app/DiffusionDet

# Comando por defecto para mantener el contenedor en ejecución
CMD ["sleep", "infinity"]
