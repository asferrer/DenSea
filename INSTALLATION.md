## DiffusionDet: Diffusion Model for Object Detection

## CUDA & CUDNN versions needed
```bash
CUDA: 10.2
CUDNN: 8.7.0
```
## Installation (conda & pip)
```bash
conda create -n DenSea python=3.7.16
conda activate DenSea
pip install -r requierements.txt
pip install -e detectron2
```

## Models
Method | Box AP (1 step) | Box AP (4 step) | Download
--- |:---:|:---:|:---:
[COCO-Res50](configs/diffdet.coco.res50.yaml) | 45.5 | 46.1 | [model](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_coco_res50.pth)
[COCO-Res101](configs/diffdet.coco.res101.yaml) | 46.6 | 46.9 | [model](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_coco_res101.pth)
[COCO-SwinBase](configs/diffdet.coco.swinbase.yaml) | 52.3 | 52.7 | [model](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_coco_swinbase.pth)
[LVIS-Res50](configs/diffdet.lvis.res50.yaml) | 30.4 | 31.8 | [model](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_lvis_res50.pth)
[LVIS-Res101](configs/diffdet.lvis.res101.yaml) | 31.9 | 32.9 | [model](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_lvis_res101.pth)
[LVIS-SwinBase](configs/diffdet.lvis.swinbase.yaml) | 40.6 | 41.9 | [model](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_lvis_swinbase.pth)


## Trainning usefull commands
```bash
python train_net_cleansea.py --num-gpus 1 --config-file configs/diffdet.coco.res50.300boxes_cleansea.yaml
```

## Testing Usefull commands
```bash
python demo.py --config-file configs/diffdet.coco.res50.300boxes.yaml --input IMG-20230329-WA0022.jpg --opts MODEL.WEIGHTS models/diffdet_coco_res50_300boxes.pth

python demo.py --config-file configs/diffdet.coco.res50.300boxes.yaml --video-input C:\Cleansea\cleansea_dataset\Videos\video_analisis\debrisVideo2.mp4 --opts MODEL.WEIGHTS models/diffdet_coco_res50_300boxes.pth

python demo.py --config-file configs/diffdet.coco.res50.300boxes.yaml --webcam --confidence-threshold 0.8 --opts MODEL.WEIGHTS models/diffdet_coco_res50_300boxes.pth
```