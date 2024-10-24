# Bitacora de entrenamientos
``tensorboard --logdir /app/DiffusionDet/Densea/output --host 0.0.0.0 --port 6006``
## Entrenamiento con swinbase tiny
1. ``nohup python train_net_cleansea.py --num-gpus 1 --config-file configs/diffdet.densea.swintinyv1.yaml --resume &> train-swintinyv1.log &``
2. ``nohup python train_net_cleansea.py --num-gpus 1 --config-file configs/diffdet.densea.res50.yaml --resume &> train-resnet50_v2.log &``
3. ``nohup python train_net_cleansea.py --num-gpus 1 --config-file configs/diffdet.densea.res101.yaml --resume &> train-resnet101_v2.log &``
4. ``nohup python train_net_cleansea.py --num-gpus 1 --config-file configs/diffdet.densea.swinsmallv1.yaml --resume &> train-swinsmallv1.log &``
4. ``nohup python train_net_cleansea.py --num-gpus 1 --config-file configs/diffdet.densea.swinbasev2.yaml --resume &> train-swinbasev2.log &``

# Comprobación de ejecucion entrenamientos
ps aux | grep train_net_cleansea.py

kill 

nohup tensorboard --logdir ./output/swintiny_densea_v1 --host 0.0.0.0 --port 6006 &

# Comandos de ejemplo de entrenamiento y ejecución
python demo.py --config-file configs/diffdet.coco.res50.300boxes.yaml --input IMG-20230329-WA0022.jpg --opts MODEL.WEIGHTS models/diffdet_coco_res50_300boxes.pth
python train_net_cleansea.py --num-gpus 1 --config-file configs/diffdet.coco.res50.300boxes_cleansea.yaml
python demo.py --config-file configs/diffdet.coco.res50.300boxes.yaml --video-input C:\Cleansea\cleansea_dataset\Videos\video_analisis\debrisVideo2.mp4 --opts MODEL.WEIGHTS models/diffdet_coco_res50_300boxes.pth
python demo.py --config-file configs/diffdet.coco.res50.300boxes.yaml --webcam --confidence-threshold 0.8 --opts MODEL.WEIGHTS models/diffdet_coco_res50_300boxes.pth

``python train_net_cleansea.py --num-gpus 1 --config-file configs/diffdet.coco.res101.cleansea.yaml --eval-only``
``python demo.py --config-file configs/diffdet.coco.res101.cleansea.yaml --video-input C:\Cleansea\cleansea_dataset\Videos\video_analisis\debrisVideo2.mp4 --output debrisVideo2_diffusiondet_1.mp4 --opts MODEL.WEIGHTS models/diffdet_cleansea_res101.``
``python demo.py --config-file configs/diffdet.coco.res101.cleansea.yaml --video-input C:\Cleansea\cleansea_dataset\Videos\video_analisis\debrisVideo_PRL1.mp4 --output debrisVideo_PRL1_diffusiondet.mp4 --opts MODEL.WEIGHTS models/diffdet_cleansea_res101.pth``

``python train_net_cleansea.py --num-gpus 1 --config-file configs/diffdet.coco.swinbase.cleansea.yaml``
