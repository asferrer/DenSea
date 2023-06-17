# Start with the official nvidia cuda image
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# Set the working directory
WORKDIR /app

# Copy everything into the container
COPY . .

# Update CUDA Keys
RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

# Update the package manager and install basic packages
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip

# Install Python 3.7.13 using pyenv
RUN apt-get update

RUN curl https://pyenv.run | bash
ENV PYENV_ROOT /root/.pyenv
ENV PATH /root/.pyenv/shims:/root/.pyenv/bin:$PATH
RUN pyenv install 3.7.13
RUN pyenv global 3.7.13

# Check python version
RUN python --version

# Install Python pip
RUN apt-get update && apt-get install -y python3-pip --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Install scikit-build before other Python dependencies
RUN pip3 install scikit-build

# Install python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install custom detectron2
RUN pip3 install --no-cache-dir -e detectron2

# Expose any ports the app is expecting in the environment
ENV PORT 8888
EXPOSE $PORT

# Set the default command to run a shell
CMD ["/bin/bash"]
