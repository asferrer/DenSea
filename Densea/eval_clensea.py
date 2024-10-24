#!/usr/bin/env python
"""
Script para evaluar un modelo entrenado de DiffusionDet y obtener métricas detalladas,
utilizando un archivo de configuración en lugar de argumentos de línea de comandos.
"""

import os
import sys
import time
import json
import logging
from collections import defaultdict

import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.engine import default_setup
from detectron2.structures import Boxes, BoxMode, pairwise_iou

from diffusiondet import add_diffusiondet_config
from diffusiondet.predictor import VisualizationDemo

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Añade el directorio de DiffusionDet al path si es necesario
sys.path.append('./diffusiondet')

def setup_cfg(config):
    # Cargar la configuración desde el archivo y los parámetros proporcionados
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    cfg.merge_from_file(config['config_file'])
    cfg.merge_from_list(config.get('opts', []))

    # Establecer el umbral de confianza
    confidence_threshold = config.get('confidence_threshold', 0.5)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold

    # Cargar los pesos del modelo entrenado
    cfg.MODEL.WEIGHTS = config['model_weights']

    # Especificar el conjunto de datos de prueba
    cfg.DATASETS.TEST = (config['dataset_name'],)

    cfg.freeze()
    default_setup(cfg, config)
    return cfg

def evaluate_model(cfg, config):
    """
    Evalúa el modelo y calcula métricas detalladas.
    """
    logger = logging.getLogger(__name__)

    # Configurar el demo de visualización
    demo = VisualizationDemo(cfg)

    # Obtener el conjunto de datos de prueba
    dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TEST[0])
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    class_names = metadata.get("thing_classes", None)
    num_classes = len(class_names)

    # Listas para almacenar las etiquetas y predicciones
    all_gt_classes = []
    all_pred_classes = []
    all_scores = []

    # Directorio de salida para visualizaciones
    output_dir = config.get('output_dir', './output/evaluation')
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Comenzando la evaluación...")

    for idx, data in enumerate(tqdm.tqdm(dataset_dicts)):
        # Leer la imagen
        img = read_image(data["file_name"], format="BGR")
        start_time = time.time()

        # Realizar predicción y obtener visualización
        predictions, visualized_output = demo.run_on_image(img)

        # Guardar visualización si se especifica el directorio de salida
        if config.get('save_visualizations', False):
            out_filename = os.path.join(output_dir, os.path.basename(data["file_name"]))
            visualized_output.save(out_filename)

        # Obtener las predicciones
        instances = predictions["instances"].to("cpu")
        pred_boxes = instances.pred_boxes.tensor.numpy()
        pred_classes = instances.pred_classes.numpy()
        pred_scores = instances.scores.numpy()

        # Obtener las anotaciones verdaderas
        gt_boxes = np.array([ann["bbox"] for ann in data["annotations"]])
        gt_classes = np.array([ann["category_id"] for ann in data["annotations"]])

        # Convertir gt_boxes de XYWH a XYXY
        gt_boxes = BoxMode.convert(gt_boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

        # Calcular IoU entre predicciones y ground truth
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            ious = pairwise_iou(Boxes(pred_boxes), Boxes(gt_boxes)).numpy()
            # Asignar predicciones a ground truths
            assigned_gt = -np.ones(len(gt_boxes), dtype=int)
            assigned_pred = -np.ones(len(pred_boxes), dtype=int)
            for i in range(len(pred_boxes)):
                max_iou = config.get('iou_thresh', 0.5)
                max_j = -1
                for j in range(len(gt_boxes)):
                    if ious[i, j] >= max_iou and assigned_gt[j] == -1:
                        max_iou = ious[i, j]
                        max_j = j
                if max_j >= 0:
                    assigned_gt[max_j] = i
                    assigned_pred[i] = max_j
        else:
            assigned_gt = -np.ones(len(gt_boxes), dtype=int)
            assigned_pred = -np.ones(len(pred_boxes), dtype=int)

        # Procesar predicciones
        for i, pred_class in enumerate(pred_classes):
            if assigned_pred[i] != -1:
                gt_class = gt_classes[assigned_pred[i]]
                all_gt_classes.append(gt_class)
                all_pred_classes.append(pred_class)
                all_scores.append(pred_scores[i])
            else:
                # Falso positivo
                all_gt_classes.append(num_classes)  # Clase desconocida para FP
                all_pred_classes.append(pred_class)
                all_scores.append(pred_scores[i])

        # Procesar ground truths no detectados (falsos negativos)
        for j, gt_class in enumerate(gt_classes):
            if assigned_gt[j] == -1:
                # Falso negativo
                all_gt_classes.append(gt_class)
                all_pred_classes.append(num_classes)  # Clase desconocida para FN
                all_scores.append(0.0)

    # Añadir clase "Desconocido" para FP y FN
    class_names_extended = class_names + ["Desconocido"]

    # Convertir a numpy arrays
    y_true = np.array(all_gt_classes)
    y_pred = np.array(all_pred_classes)
    y_scores = np.array(all_scores)

    # Calcular la matriz de confusión
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes + 1))

    # Guardar la matriz de confusión
    plot_confusion_matrix(cm, class_names_extended, output_dir)

    # Calcular precisión, recall y F1 score por clase
    precision, recall, f1_score, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(num_classes)
    )

    # Crear un reporte
    report = {}
    for idx, class_name in enumerate(class_names):
        report[class_name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1_score": float(f1_score[idx]),
            "support": int(support[idx]),
        }

    # Calcular métricas globales
    avg_precision = precision.mean()
    avg_recall = recall.mean()
    avg_f1 = f1_score.mean()

    report["global_metrics"] = {
        "average_precision": float(avg_precision),
        "average_recall": float(avg_recall),
        "average_f1_score": float(avg_f1),
    }

    # Guardar el reporte en un archivo JSON
    report_path = os.path.join(output_dir, 'classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    logger.info(f"Reporte de clasificación guardado en {report_path}")

    # Mostrar métricas globales en el logger
    logger.info(f"Métricas globales - Precisión: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1 Score: {avg_f1:.4f}")

def plot_confusion_matrix(cm, class_names, output_dir):
    """
    Genera y guarda la matriz de confusión.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.xticks(rotation=45)
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    logging.getLogger(__name__).info(f"Matriz de confusión guardada en {cm_path}")

def main():
    # Cargar el archivo de configuración
    with open('evaluation_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Configurar el logger
    logger = setup_logger()
    logger.info("Iniciando la evaluación del modelo...")

    cfg = setup_cfg(config)

    # Registrar los conjuntos de datos personalizados
    from detectron2.data.datasets import register_coco_instances

    data_root = config['data_root']

    # Registrar datasets
    dataset_name = config['dataset_name']
    dataset_ann = os.path.join(data_root, f"{dataset_name}/annotations_bbox.json")
    dataset_img = os.path.join(data_root, dataset_name)
    register_coco_instances(
        dataset_name,
        {},
        dataset_ann,
        dataset_img
    )

    # Definir las clases
    classes = config['classes']

    # Actualizar el Metadata
    MetadataCatalog.get(dataset_name).set(thing_classes=classes)

    # Evaluar el modelo
    evaluate_model(cfg, config)

    # Imprimir resultados finales
    logger.info("Evaluación completada. Resultados guardados en el directorio de salida.")

if __name__ == "__main__":
    main()
