#!/usr/bin/env python
"""
Script para evaluar un modelo entrenado de DiffusionDet y obtener métricas detalladas,
almacenando las inferencias y gráficas en imágenes, y mostrando etiquetas reales vs predicciones.
"""

import os
import sys
import time
import json
import logging
from collections import defaultdict

import numpy as np
import cv2
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
from detectron2.utils.visualizer import Visualizer
from diffusiondet.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer

from diffusiondet import add_diffusiondet_config
from diffusiondet.predictor import VisualizationDemo

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Añade el directorio de DiffusionDet al path si es necesario
sys.path.append('./diffusiondet')

def setup_cfg(config):
    # Cargar la configuración desde el archivo y los parámetros proporcionados
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
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

    # Lista para almacenar métricas por imagen
    per_image_metrics = []

    # Directorio de salida para resultados
    output_dir = config.get('output_dir', './output/evaluation')
    os.makedirs(output_dir, exist_ok=True)

    # Directorio para guardar las inferencias de las imágenes
    inference_dir = os.path.join(output_dir, 'inferences')
    os.makedirs(inference_dir, exist_ok=True)

    logger.info("Comenzando la evaluación...")

    for idx, data in enumerate(tqdm.tqdm(dataset_dicts)):
        # Leer la imagen
        img = read_image(data["file_name"], format="BGR")
        start_time = time.time()

        # Visualizar las anotaciones de ground truth
        visualizer_gt = Visualizer(img[:, :, ::-1], metadata=metadata)
        vis_gt = visualizer_gt.draw_dataset_dict(data)
        gt_image = vis_gt.get_image()[:, :, ::-1]  # Convertir de RGB a BGR

        # Realizar predicción y obtener visualización
        predictions, visualized_output = demo.run_on_image(img)

        # Convertir visualized_output a imagen
        pred_image = visualized_output.get_image()[:, :, ::-1]  # BGR format

        # Combinar las imágenes lado a lado
        combined_image = np.concatenate((gt_image, pred_image), axis=1)

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

        # Inicializar métricas por imagen
        num_true_positives = 0
        num_false_positives = 0
        num_false_negatives = 0
        num_correct_classes = 0
        num_incorrect_classes = 0

        # Procesar predicciones
        for i, pred_class in enumerate(pred_classes):
            if assigned_pred[i] != -1:
                gt_class = gt_classes[assigned_pred[i]]
                if pred_class == gt_class:
                    num_true_positives += 1
                    num_correct_classes += 1
                else:
                    num_true_positives += 1  # Detección correcta pero clase incorrecta
                    num_incorrect_classes += 1
                all_gt_classes.append(gt_class)
                all_pred_classes.append(pred_class)
                all_scores.append(pred_scores[i])
            else:
                # Falso positivo
                num_false_positives += 1
                all_gt_classes.append(num_classes)  # Clase desconocida para FP
                all_pred_classes.append(pred_class)
                all_scores.append(pred_scores[i])

        # Procesar ground truths no detectados (falsos negativos)
        for j, gt_class in enumerate(gt_classes):
            if assigned_gt[j] == -1:
                # Falso negativo
                num_false_negatives += 1
                all_gt_classes.append(gt_class)
                all_pred_classes.append(num_classes)  # Clase desconocida para FN
                all_scores.append(0.0)

        # Almacenar métricas por imagen
        image_metrics = {
            'image_id': data.get('image_id', idx),
            'file_name': data['file_name'],
            'num_ground_truths': len(gt_boxes),
            'num_predictions': len(pred_boxes),
            'num_true_positives': num_true_positives,
            'num_false_positives': num_false_positives,
            'num_false_negatives': num_false_negatives,
            'num_correct_classes': num_correct_classes,
            'num_incorrect_classes': num_incorrect_classes
        }

        per_image_metrics.append(image_metrics)

        # Añadir texto con las métricas al final de la imagen
        metrics_text = f"TP: {num_true_positives}, FP: {num_false_positives}, FN: {num_false_negatives}, " \
                       f"Correct Class: {num_correct_classes}, Incorrect Class: {num_incorrect_classes}"

        # Definir el tamaño y posición del texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_size, _ = cv2.getTextSize(metrics_text, font, font_scale, font_thickness)
        text_x = 10
        text_y = combined_image.shape[0] + text_size[1] + 10

        # Crear una imagen para el texto
        text_image = np.zeros((text_size[1] + 20, combined_image.shape[1], 3), dtype=np.uint8)

        # Escribir el texto en la imagen del texto
        cv2.putText(text_image, metrics_text, (text_x, text_size[1] + 5), font, font_scale, (255, 255, 255), font_thickness)

        # Combinar la imagen del texto con la imagen combinada
        combined_with_text = np.vstack((combined_image, text_image))

        # Guardar la imagen final en la carpeta de inferencias
        out_filename = os.path.join(inference_dir, os.path.basename(data["file_name"]))
        cv2.imwrite(out_filename, combined_with_text)

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

    # Guardar las métricas por imagen
    metrics_path = os.path.join(output_dir, 'per_image_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(per_image_metrics, f, indent=4)
    logger.info(f"Métricas por imagen guardadas en {metrics_path}")

    # Mostrar métricas globales en el logger
    logger.info(f"Métricas globales - Precisión: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1 Score: {avg_f1:.4f}")

    # Generar gráficas adicionales
    generate_additional_plots(precision, recall, f1_score, class_names, output_dir)

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

def generate_additional_plots(precision, recall, f1_score, class_names, output_dir):
    """
    Genera y guarda gráficas de precisión, recall y F1 score por clase.
    """
    indices = np.arange(len(class_names))

    # Gráfica de Precisión por clase
    plt.figure(figsize=(12, 6))
    plt.bar(indices, precision, color='skyblue')
    plt.xticks(indices, class_names, rotation=45)
    plt.ylabel('Precisión')
    plt.title('Precisión por Clase')
    plt.tight_layout()
    precision_path = os.path.join(output_dir, 'precision_per_class.png')
    plt.savefig(precision_path)
    plt.close()

    # Gráfica de Recall por clase
    plt.figure(figsize=(12, 6))
    plt.bar(indices, recall, color='lightgreen')
    plt.xticks(indices, class_names, rotation=45)
    plt.ylabel('Recall')
    plt.title('Recall por Clase')
    plt.tight_layout()
    recall_path = os.path.join(output_dir, 'recall_per_class.png')
    plt.savefig(recall_path)
    plt.close()

    # Gráfica de F1 Score por clase
    plt.figure(figsize=(12, 6))
    plt.bar(indices, f1_score, color='salmon')
    plt.xticks(indices, class_names, rotation=45)
    plt.ylabel('F1 Score')
    plt.title('F1 Score por Clase')
    plt.tight_layout()
    f1_path = os.path.join(output_dir, 'f1_score_per_class.png')
    plt.savefig(f1_path)
    plt.close()

    logging.getLogger(__name__).info("Gráficas de precisión, recall y F1 score guardadas en el directorio de salida.")

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

    if config['grouped']: dataset_ann = os.path.join(data_root, f"{dataset_name}/annotations_bbox_grouped.json")
    else: dataset_ann = os.path.join(data_root, f"{dataset_name}/annotations_bbox.json")
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
