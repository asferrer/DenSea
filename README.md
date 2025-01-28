# DiffusionDet for Marine Debris Detection

![CUDA](https://img.shields.io/badge/CUDA-12.1.0-orange)
![CUDNN](https://img.shields.io/badge/CUDNN-8.7.0-orange)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Marine Debris Detection Goals](#marine-debris-detection-goals)
- [Installation](#installation)
  - [Using DevContainer](#using-devcontainer)
  - [Using Conda & Pip](#using-conda--pip)
- [Models](#models)
- [Training Commands](#training-commands)
- [Testing Commands](#testing-commands)
- [Docker Setup](#docker-setup)
  - [Dockerfile](#dockerfile)
  - [docker-compose.yml](#docker-composeyml)
- [Contributing](#contributing)
- [License](#license)
- [Citing](#citing)
- [Contact](#contact)

## Project Overview

Project DenSea leverages **DiffusionDet**, a cutting-edge object detection model, for identifying and tracking **marine debris** in underwater environments. It utilizes diffusion-based methods for enhanced accuracy and is specifically tailored to address the challenges of detecting objects in diverse underwater conditions, such as varying light levels, water turbidity, and debris occlusion.

The implementation is based on **DiffusionDet v0.1** and the latest version of **Detectron2**, ensuring compatibility with modern datasets and hardware configurations.

## Features

- **Marine Debris Detection:** Detects objects like plastics, metals, tires, and other waste commonly found on the seabed.
- **Advanced Diffusion Techniques:** Employs diffusion-based methods to improve detection accuracy in complex environments.
- **Dataset Compatibility:** Supports COCO and LVIS datasets for training and testing.
- **Dockerized Development:** Includes a DevContainer setup for seamless environment configuration.
- **Visualization Tools:** Integrated with TensorBoard for monitoring training performance and metrics.
- **Streamlit Interface:** Offers an interactive web interface for real-time predictions and dataset exploration.

## Marine Debris Detection Goals

The primary goal of this application is to assist researchers and environmental organizations in:

- **Monitoring marine pollution:** Identifying types and quantities of debris across different underwater regions.
- **Developing cleanup strategies:** Using detected patterns to plan effective removal of underwater waste.
- **Raising awareness:** Visualizing marine debris data to educate stakeholders and promote ocean conservation.

## Installation

### Using DevContainer

1. Clone the repository and navigate to the project directory.
  ```bash
  git clone https://github.com/asferrer/DenSea.git
  cd DenSea
  ```
2. Ensure Docker and Docker Compose are installed on your system.
3. Build and run the DevContainer using the provided `docker-compose.yml` file. This will set up all necessary dependencies and libraries in a consistent environment.
```bash
  docker-compose up -d
  ```
4. Once the container is running, access the Streamlit interface at `http://localhost:8501`.

### Using Conda & Pip

1. Clone the repository and navigate to the project directory.
```bash
  git clone https://github.com/asferrer/DenSea.git
  cd DenSea
  ```
2. Create a virtual environment using Conda with Python 3.10.
```bash
conda create -n DenSea python=3.10
conda activate DenSea
```
3. Install the `detectron2` and `diffusiondet` packages for object detection functionality.
```bash
pip install --upgrade pip
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install timm
```

## Models

Below is a summary of the supported models, their performance metrics, and links to download the pre-trained weights:

| Method          | Box AP (1 step) | Box AP (4 step) | Download Link                                                                 |
|------------------|-----------------|-----------------|-------------------------------------------------------------------------------|
| COCO-Res50       | 45.5           | 46.1           | [model](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_coco_res50.pth) |
| COCO-Res101      | 46.6           | 46.9           | [model](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_coco_res101.pth) |
| COCO-SwinBase    | 52.3           | 52.7           | [model](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_coco_swinbase.pth) |
| LVIS-Res50       | 30.4           | 31.8           | [model](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_lvis_res50.pth) |
| LVIS-Res101      | 31.9           | 32.9           | [model](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_lvis_res101.pth) |
| LVIS-SwinBase    | 40.6           | 41.9           | [model](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_lvis_swinbase.pth) |

## Training Commands

To train the model for marine debris detection, use the appropriate configuration file for your dataset. Specify the number of GPUs and the configuration YAML file for training.
```bash
python train_net_cleansea.py --num-gpus 1 --config-file configs/diffdet.coco.res50.300boxes_cleansea.yaml
```

## Testing Commands

The application supports testing on:

- **Images:** Specify the path to an input image for detection.
```bash
python demo.py --config-file configs/diffdet.coco.res50.300boxes.yaml --input IMG-20230329-WA0022.jpg --opts MODEL.WEIGHTS models/diffdet_coco_res50_300boxes.pth
```
- **Videos:** Specify the path to a video file for detection over frames.
```bash
python demo.py --config-file configs/diffdet.coco.res50.300boxes.yaml --video-input C:\Cleansea\cleansea_dataset\Videos\video_analisis\debrisVideo2.mp4 --opts MODEL.WEIGHTS models/diffdet_coco_res50_300boxes.pth
```
- **Live Webcam Feed:** Use your webcam to detect debris in real-time.
```bash
python demo.py --config-file configs/diffdet.coco.res50.300boxes.yaml --webcam --confidence-threshold 0.8 --opts MODEL.WEIGHTS models/diffdet_coco_res50_300boxes.pth
```

For each testing scenario, ensure the path to the trained model weights is provided.

## Docker Setup

### Dockerfile

The Dockerfile is based on `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04` and sets up the required environment, including Python dependencies and GPU support.

### docker-compose.yml

The `docker-compose.yml` file configures the DevContainer for the project. It includes:

- Volume mounting for the source code.
- Port mappings for TensorBoard and Streamlit.
- Resource limits to optimize memory and CPU usage.
- Shared memory size increase to support large datasets.

## Contributing

We welcome contributions to improve the project. Follow these steps to contribute:

1. Fork the repository.
2. Clone your fork and create a new branch for your feature or bug fix.
3. Make your changes and commit them with clear, concise messages.
4. Push your branch to your fork and open a pull request.

## License

This project is licensed under the MIT License.

## Citing

If you use DiffusionDet in your research or wish to refer to the results published here, please use the following BibTeX entries.

```BibTeX
@article{SANCHEZFERRER2023154,
      title = {An experimental study on marine debris location and recognition using object detection},
      author = {Alejandro Sánchez-Ferrer and Jose J. Valero-Mas and Antonio Javier Gallego and Jorge Calvo-Zaragoza},
      journal = {Pattern Recognition Letters},
      year = {2023},
      doi = {https://doi.org/10.1016/j.patrec.2022.12.019},
      url = {https://www.sciencedirect.com/science/article/pii/S0167865522003889},
}
```
```BibTeX
@InProceedings{10.1007/978-3-031-04881-4_49,
      title="The CleanSea Set: A Benchmark Corpus for Underwater Debris Detection and Recognition",
      author="S{\'a}nchez-Ferrer, Alejandro and Gallego, Antonio Javier and Valero-Mas, Jose J. and Calvo-Zaragoza, Jorge",
      booktitle="Pattern Recognition and Image Analysis",
      year="2022",
      publisher="Springer International Publishing",
}

```
```BibTeX
@article{chen2022diffusiondet,
      title={DiffusionDet: Diffusion Model for Object Detection},
      author={Chen, Shoufa and Sun, Peize and Song, Yibing and Luo, Ping},
      journal={arXiv preprint arXiv:2211.09788},
      year={2022}
}
```

## Contact

For inquiries or support, contact:

**Alejandro Sanchez Ferrer**  
Email: asanc.tech@gmail.com  
GitHub: [asferrer](https://github.com/asferrer)
