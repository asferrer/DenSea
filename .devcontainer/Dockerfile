FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# Establecer variables de entorno para evitar interacciones durante la instalación
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Instalar dependencias del sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        libgl1 \
        libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Crear un enlace simbólico para 'python'
RUN ln -s /usr/bin/python3 /usr/bin/python

# Actualizar pip
RUN pip install --no-cache-dir --upgrade pip

# Instalar detectron2 desde el repositorio oficial
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

# Instalar otras dependencias necesarias
RUN pip install --no-cache-dir \
        timm \
        opencv-python \
        seaborn \
        scikit-learn \
        streamlit \
        albumentations \
        iterative-stratification \
        onnxruntime-gpu \
        onnx \
        onnx-simplifier

# Establecer el directorio de trabajo
WORKDIR /app/DiffusionDet

# Comando por defecto para mantener el contenedor en ejecución
CMD ["sleep", "infinity"]
