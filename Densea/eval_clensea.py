git a#!/usr/bin/env python
"""
Script de evaluación avanzado e independiente para DiffusionDet.
Calcula:
- Métricas detalladas personalizadas (Matriz de confusión, P/R/F1 por clase).
- Métricas COCO estándar (mAP) usando COCOEvaluator.
- Tiempo de inferencia.
Genera:
- Visualizaciones de inferencia estándar (GT vs Pred).
- Visualizaciones de Top-K errores (FP, FN).
- Plots de distribución de scores e IoU.
- Reporte exhaustivo en texto y JSON.

Modificaciones v4.2 (matching_fix):
- Usar metadata.thing_classes como fuente de verdad para nombres/número de clases en métricas personalizadas.
- Validación más robusta de IDs de clase.
- Logging de depuración mejorado para el proceso de matching (visible con --log-level DEBUG).
- Corrección previa de OpenCV readonly array mantenida.
"""

import os
import sys
import logging
import copy
import yaml
import numpy as np
import cv2
import torch
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
import time 
from collections import defaultdict

from detectron2.config import get_cfg, CfgNode
from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog,
    build_detection_test_loader,
    detection_utils as utils,
    transforms as T
)
from detectron2.data.detection_utils import read_image
from detectron2.structures import Boxes, BoxMode, pairwise_iou, Instances
from detectron2.utils.visualizer import Visualizer, VisImage
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import COCOEvaluator

from diffusiondet import add_diffusiondet_config
from diffusiondet.util.model_ema import (
    add_model_ema_configs,
    may_build_model_ema,
    may_get_ema_checkpointer,
    apply_model_ema_and_restore,
    EMADetectionCheckpointer
)

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

logger = logging.getLogger("eval_script_advanced")

# --- Clases y Funciones Helper para Análisis Avanzado ---
class ErrorLogger:
    def __init__(self, top_k=10):
        self.top_k = top_k
        self.false_positives = [] # (score, image_path, pred_box, pred_class_idx, image_id, gt_class_idx_if_misclassified)
        self.false_negatives = [] # (image_path, gt_box, gt_class_idx, image_id)

    def add_fp(self, score, image_path, pred_box, pred_class_idx, image_id, gt_class_idx_if_misclassified=None):
        self.false_positives.append((score, image_path, pred_box, pred_class_idx, image_id, gt_class_idx_if_misclassified))

    def add_fn(self, image_path, gt_box, gt_class_idx, image_id):
        self.false_negatives.append((image_path, gt_box, gt_class_idx, image_id))

    def get_top_k_fps(self):
        self.false_positives.sort(key=lambda x: x[0], reverse=True)
        return self.false_positives[:self.top_k]

    def get_top_k_fns(self):
        return self.false_negatives[:self.top_k]

def visualize_errors(errors, error_type, output_dir, class_names_list, max_vis=10):
    if not errors:
        logger.info(f"No se encontraron errores de tipo '{error_type}' para visualizar.")
        return

    error_vis_dir = os.path.join(output_dir, f"top_{error_type}_visualizations")
    os.makedirs(error_vis_dir, exist_ok=True)
    logger.info(f"Guardando visualizaciones de Top-{error_type} en: {error_vis_dir}")

    for i, error_data in enumerate(errors[:max_vis]):
        try:
            if error_type == "false_positives":
                score, image_path, box_coords, pred_class_idx, img_id, gt_class_idx_if_misclassified = error_data
                pred_class_name = class_names_list[pred_class_idx] if pred_class_idx < len(class_names_list) else "OOR_PredCls" # Out Of Range
                text = f"FP: {pred_class_name} (S: {score:.2f})"
                if gt_class_idx_if_misclassified is not None:
                    gt_class_name = class_names_list[gt_class_idx_if_misclassified] if gt_class_idx_if_misclassified < len(class_names_list) else "OOR_GTCls"
                    text += f"\n(GT: {gt_class_name})"
                box_color = (255, 0, 0) 
            elif error_type == "false_negatives":
                image_path, box_coords, gt_class_idx, img_id = error_data
                gt_class_name = class_names_list[gt_class_idx] if gt_class_idx < len(class_names_list) else "OOR_GTCls"
                text = f"FN: {gt_class_name}"
                box_color = (0, 0, 255) 
            else:
                continue

            img_bgr = read_image(image_path, format="BGR")
            if img_bgr is None:
                logger.warning(f"No se pudo leer la imagen {image_path} para visualizar error.")
                continue
            
            img_bgr = img_bgr.copy() 

            x1, y1, x2, y2 = map(int, box_coords)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), box_color, 2)
            
            # For multiline text
            y_text = y1 - 10
            for line_idx, line in enumerate(text.split('\n')):
                cv2.putText(img_bgr, line, (x1, y_text - (line_idx * 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
            vis_path_full = os.path.join(error_vis_dir, f"{error_type}_{img_id}_{i}.png")
            cv2.imwrite(vis_path_full, img_bgr)

            margin = 10 
            crop_x1 = max(0, x1 - margin)
            crop_y1 = max(0, y1 - margin)
            crop_x2 = min(img_bgr.shape[1], x2 + margin)
            crop_y2 = min(img_bgr.shape[0], y2 + margin)
            
            if crop_y2 > crop_y1 and crop_x2 > crop_x1 :
                error_crop = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
                crop_path = os.path.join(error_vis_dir, f"{error_type}_crop_{img_id}_{i}.png")
                cv2.imwrite(crop_path, error_crop)

        except Exception as e:
            logger.error(f"Error visualizando {error_type} para {image_path}: {e}", exc_info=True)


def plot_distributions(data_dict, title, xlabel, output_path, bins=30):
    # (Sin cambios)
    plt.figure(figsize=(12, 7))
    for label, values in data_dict.items():
        if values:
            sns.histplot(values, label=label, kde=True, bins=bins, stat="density", common_norm=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Densidad")
    if any(data_dict.values()): 
        plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Plot de distribución guardado en: {output_path}")

class EvaluationDatasetMapper:
    # (Sin cambios)
    def __init__(self, cfg: CfgNode, is_train: bool = False):
        self.img_format = cfg.INPUT.FORMAT
        self.augmentations = T.AugmentationList([
            T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, "choice")
        ])
    def __call__(self, dataset_dict: dict) -> dict:
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(aug_input.image.transpose(2, 0, 1)))
        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if anno.get("bbox_mode") != BoxMode.XYXY_ABS:
                    anno["bbox"] = BoxMode.convert(anno["bbox"], anno["bbox_mode"], BoxMode.XYXY_ABS)
                    anno["bbox_mode"] = BoxMode.XYXY_ABS
        return dataset_dict

def load_evaluation_config(config_path: str) -> dict:
    # (Sin cambios)
    if not os.path.exists(config_path): sys.exit(f"Config file '{config_path}' not found.")
    with open(config_path, 'r') as f: eval_config = yaml.safe_load(f)
    # ... (setdefault calls como antes)
    eval_config.setdefault('confidence_threshold', 0.05)
    eval_config.setdefault('iou_thresh_metrics', 0.5)
    eval_config.setdefault('use_absolute_paths_in_json', True)
    eval_config.setdefault('data_root', '.')
    eval_config.setdefault('opts', [])
    eval_config.setdefault('visualization_scale', 0.8)
    eval_config.setdefault('use_ema_weights', False)
    eval_config.setdefault('analyze_top_k_errors', 10) 
    eval_config.setdefault('plot_score_distributions', True)
    eval_config.setdefault('plot_iou_distribution', True)
    return eval_config

def setup_detectron2_cfg(eval_config: dict) -> CfgNode:
    # (Sin cambios)
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    # ... (merge de configs y pesos como antes)
    model_config_file_path = eval_config['model_config_file']
    if not os.path.isabs(model_config_file_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_config_file_path = os.path.join(script_dir, model_config_file_path)
    if not os.path.exists(model_config_file_path):
        sys.exit(f"Model config file '{model_config_file_path}' not found.")
    cfg.merge_from_file(model_config_file_path)
    cfg.merge_from_list(eval_config['opts'])
    
    model_weights_path = eval_config['model_weights']
    if not os.path.isabs(model_weights_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_weights_path = os.path.join(script_dir, model_weights_path)
    if not os.path.exists(model_weights_path):
        sys.exit(f"Model weights file '{model_weights_path}' not found.")
    cfg.MODEL.WEIGHTS = model_weights_path
    cfg.DATASETS.TEST = (eval_config['dataset_name'],)
    cfg.defrost()
    cfg.MODEL_EMA.ENABLED = eval_config['use_ema_weights']
    cfg.MODEL_EMA.LOAD_EMA_WEIGHTS = eval_config['use_ema_weights']
    cfg.freeze()
    return cfg

def register_evaluation_dataset(eval_config: dict):
    # (Sin cambios significativos, ya usaba metadata)
    dataset_name_eval = eval_config['dataset_name']
    annotation_file_path = eval_config['annotation_file']
    # ... (manejo de rutas como antes)
    if not os.path.isabs(annotation_file_path):
        data_root_abs = os.path.abspath(eval_config['data_root'])
        annotation_file_path = os.path.join(data_root_abs, annotation_file_path)
    if not os.path.exists(annotation_file_path):
        sys.exit(f"Annotation file '{annotation_file_path}' not found.")

    image_root_for_registration = ""
    if not eval_config.get('use_absolute_paths_in_json', True):
        image_root_for_registration = os.path.abspath(eval_config['data_root'])
    
    if dataset_name_eval in DatasetCatalog.list(): DatasetCatalog.remove(dataset_name_eval)
    if dataset_name_eval in MetadataCatalog.list(): MetadataCatalog.remove(dataset_name_eval)
    
    register_coco_instances(
        name=dataset_name_eval, metadata={},
        json_file=annotation_file_path, image_root=image_root_for_registration
    )
    metadata_dataset = MetadataCatalog.get(dataset_name_eval)
    
    # <--- REFINEMENT: Usar metadata.thing_classes como fuente principal si está disponible
    if not hasattr(metadata_dataset, 'thing_classes') or not metadata_dataset.thing_classes:
        logger.warning(f"El JSON de anotaciones no contenía 'categories' o estaba vacía. "
                       f"Se usarán las clases de 'evaluation_config.yaml': {eval_config['classes']}")
        # Esto es un fallback, idealmente el JSON debería tener las clases.
        metadata_dataset.thing_classes = eval_config['classes']
    else:
        logger.info(f"Clases cargadas desde el JSON (vía MetadataCatalog): {metadata_dataset.thing_classes}")
        if eval_config.get('classes') and \
           set(metadata_dataset.thing_classes) != set(eval_config['classes']):
            logger.warning(f"¡Discrepancia de nombres de clases! YAML: {eval_config['classes']}, JSON: {metadata_dataset.thing_classes}. "
                           "Se priorizarán las del JSON/MetadataCatalog para métricas y análisis.")
    
    # Verificar que thing_classes no esté vacío
    if not metadata_dataset.thing_classes:
        logger.error("Error crítico: No se pudieron determinar las clases del dataset (metadata.thing_classes está vacío). "
                     "Verifica tu archivo JSON y/o la sección 'classes' en evaluation_config.yaml.")
        sys.exit(1)

    logger.info(f"Dataset '{dataset_name_eval}' registrado. Clases de referencia para métricas: {metadata_dataset.thing_classes}")
    
    if hasattr(metadata_dataset, 'thing_dataset_id_to_contiguous_id'):
        logger.info(f"Mapeo de IDs (original a contiguo) del dataset '{dataset_name_eval}': {metadata_dataset.thing_dataset_id_to_contiguous_id}")
    else:
        logger.warning(f"El dataset '{dataset_name_eval}' no tiene 'thing_dataset_id_to_contiguous_id'. "
                       "Se asume que los category_id en el JSON ya son contiguos (0 a N-1).")


def build_and_load_model(cfg: CfgNode, eval_config: dict):
    # (Sin cambios)
    model = build_model(cfg)
    # ... (lógica EMA y carga de checkpoint como antes)
    kwargs_checkpointer = {}
    if cfg.MODEL_EMA.ENABLED: 
        may_build_model_ema(cfg, model) 
        kwargs_checkpointer = may_get_ema_checkpointer(cfg, model) 
        checkpointer = EMADetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs_checkpointer)
    else:
        checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=False) 
    use_ema_for_inference_flag = False
    if cfg.MODEL_EMA.ENABLED: 
        if hasattr(model, 'ema_state') and model.ema_state and \
           hasattr(model.ema_state, 'has_inited') and model.ema_state.has_inited() and \
           hasattr(model.ema_state, 'state') and model.ema_state.state:
            use_ema_for_inference_flag = True 
    model.eval() 
    return model, use_ema_for_inference_flag

def resize_to_match_height(img_to_resize, target_height):
    # (Sin cambios)
    h, w = img_to_resize.shape[:2]
    if h == target_height: return img_to_resize
    return cv2.resize(img_to_resize, (int(target_height * w / h), target_height), interpolation=cv2.INTER_AREA)

@torch.no_grad() 
def evaluate_model_aligned(d2_cfg: CfgNode, eval_config: dict):
    model, use_ema_for_inference_flag = build_and_load_model(d2_cfg, eval_config)
    dataset_name_eval = d2_cfg.DATASETS.TEST[0]
    mapper = EvaluationDatasetMapper(d2_cfg, is_train=False)
    data_loader = build_detection_test_loader(d2_cfg, dataset_name_eval, mapper=mapper)
    
    metadata = MetadataCatalog.get(dataset_name_eval)
    
    # <--- REFINEMENT: Usar metadata.thing_classes como la fuente de verdad para métricas personalizadas
    # Esto asegura que los IDs contiguos (0 a N-1) se alineen con los nombres de clase.
    class_names_reference = metadata.thing_classes
    num_classes_reference = len(class_names_reference)
    logger.info(f"Métricas personalizadas y análisis de errores usarán {num_classes_reference} clases: {class_names_reference}")

    id_mapping = metadata.get("thing_dataset_id_to_contiguous_id", None)
    if id_mapping:
        logger.info(f"Usando mapeo de ID de GT: {id_mapping}")
    else:
        logger.info("No se encontró mapeo de ID de GT; se asume que los IDs de GT ya son contiguos 0-(N-1).")

    output_dir_abs = os.path.abspath(eval_config['output_dir'])
    # ... (creación de directorios como antes)
    os.makedirs(output_dir_abs, exist_ok=True)
    inference_images_dir = os.path.join(output_dir_abs, 'inference_visualizations_standard')
    os.makedirs(inference_images_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir_abs, 'plots_and_analysis')
    os.makedirs(plots_dir, exist_ok=True)


    # ... (Inicialización COCOEvaluator, ErrorLogger, listas de scores/IoUs, contadores TP/FP/FN como antes)
    oks_sigmas = d2_cfg.TEST.get("KEYPOINT_OKS_SIGMAS", [])
    coco_evaluator = COCOEvaluator(dataset_name_eval, output_dir=output_dir_abs, use_fast_impl=True, kpt_oks_sigmas=oks_sigmas)
    coco_evaluator.reset()

    error_analyzer = ErrorLogger(top_k=eval_config['analyze_top_k_errors'])
    scores_tp, scores_fp_cls, scores_fp_bg, ious_tp = [], [], [], []
    y_true_global, y_pred_global = [], []
    
    # Usar num_classes_reference para inicializar estos arrays
    tp_per_class = np.zeros(num_classes_reference, dtype=int)
    fp_cls_per_class = np.zeros(num_classes_reference, dtype=int)
    fp_bg_per_class = np.zeros(num_classes_reference, dtype=int)
    fn_per_class = np.zeros(num_classes_reference, dtype=int)

    total_inference_time, num_images_processed, num_batches_processed = 0, 0, 0
    eval_context = apply_model_ema_and_restore(model) if use_ema_for_inference_flag else torch.no_grad()
    
    with eval_context:
        for batch_idx, batch_inputs_from_loader in enumerate(tqdm.tqdm(data_loader, desc=f"Evaluando {dataset_name_eval}")):
            # ... (medición de tiempo y predicción del modelo como antes) ...
            start_time_batch = time.time()
            try:
                raw_model_predictions_list = model(batch_inputs_from_loader)
            except Exception as e:
                logger.error(f"Error en predicción batch {batch_idx}: {e}. Saltando.")
                continue
            total_inference_time += (time.time() - start_time_batch)
            num_batches_processed += 1
            num_images_processed += len(batch_inputs_from_loader)

            try:
                coco_evaluator.process(batch_inputs_from_loader, raw_model_predictions_list)
            except Exception as e:
                logger.error(f"Error procesando batch {batch_idx} con COCOEvaluator: {e}")

            for i, data_dict_from_loader in enumerate(batch_inputs_from_loader):
                file_path = data_dict_from_loader["file_name"]
                img_id_str = str(data_dict_from_loader.get('image_id', os.path.splitext(os.path.basename(file_path))[0]))
                logger.debug(f"\n--- Procesando imagen: {img_id_str} ({file_path}) ---")

                pred_output_single_image = raw_model_predictions_list[i]
                pred_instances_raw_cpu = pred_output_single_image["instances"].to("cpu")
                
                confidence_threshold_metrics = eval_config['confidence_threshold']
                keep = pred_instances_raw_cpu.scores >= confidence_threshold_metrics
                pred_instances_filtered = pred_instances_raw_cpu[keep]
                logger.debug(f"  Predicciones raw: {len(pred_instances_raw_cpu)}, Filtradas (conf>={confidence_threshold_metrics}): {len(pred_instances_filtered)}")

                # ... (Visualización estándar GT vs Pred como antes, usando img_id_str) ...
                img_bgr_for_vis_read = read_image(file_path, format="BGR")
                if img_bgr_for_vis_read is not None:
                    img_bgr_for_vis = img_bgr_for_vis_read.copy() 
                    img_rgb_gt = img_bgr_for_vis[:, :, ::-1].copy() 
                    v_gt = Visualizer(img_rgb_gt, metadata=metadata, scale=eval_config['visualization_scale'])
                    gt_vis_obj = v_gt.draw_dataset_dict(copy.deepcopy(data_dict_from_loader))
                    gt_img_final_rgb = gt_vis_obj.get_image()
                    img_rgb_pred = img_bgr_for_vis[:, :, ::-1].copy() 
                    v_pred = Visualizer(img_rgb_pred, metadata=metadata, scale=eval_config['visualization_scale'])
                    pred_vis_obj = v_pred.draw_instance_predictions(pred_instances_filtered)
                    pred_img_final_rgb = pred_vis_obj.get_image()
                    h_gt = gt_img_final_rgb.shape[0]
                    pred_img_final_rgb_resized = resize_to_match_height(pred_img_final_rgb, h_gt)
                    combined_bgr = np.concatenate((gt_img_final_rgb[:, :, ::-1], pred_img_final_rgb_resized[:, :, ::-1]), axis=1)
                    cv2.imwrite(os.path.join(inference_images_dir, f"eval_{img_id_str}.png"), combined_bgr)


                gt_annotations_raw = data_dict_from_loader.get("annotations", [])
                processed_gt_annotations = []
                logger.debug(f"  GTs raw para {img_id_str}: {len(gt_annotations_raw)}")
                for ann_idx, ann_raw in enumerate(gt_annotations_raw):
                    ann_copy = copy.deepcopy(ann_raw)
                    original_gt_cat_id = ann_copy["category_id"]
                    contiguous_gt_cat_id = -1

                    if id_mapping:
                        contiguous_gt_cat_id = id_mapping.get(original_gt_cat_id)
                        if contiguous_gt_cat_id is None:
                            logger.warning(f"    GT Anno {ann_idx} (img {img_id_str}): ID original {original_gt_cat_id} no encontrado en mapeo {id_mapping}. Ignorando para métricas personalizadas.")
                            continue
                    else: # No hay mapeo, asumir que el ID original ya es contiguo
                        contiguous_gt_cat_id = original_gt_cat_id
                    
                    # Validar ID contiguo contra el número de clases de referencia
                    if not (0 <= contiguous_gt_cat_id < num_classes_reference):
                        logger.warning(f"    GT Anno {ann_idx} (img {img_id_str}): ID contiguo {contiguous_gt_cat_id} (original: {original_gt_cat_id}) "
                                       f"está fuera de rango [0, {num_classes_reference-1}]. Ignorando para métricas personalizadas.")
                        continue
                    
                    ann_copy["category_id"] = contiguous_gt_cat_id # Ahora es el ID contiguo validado
                    processed_gt_annotations.append(ann_copy)
                logger.debug(f"  GTs procesados (con ID contiguo y validado) para {img_id_str}: {len(processed_gt_annotations)}")


                img_y_true, img_y_pred = [], []
                # <--- REFINEMENT: Usar num_classes_reference
                background_or_missed_label = num_classes_reference 

                if not processed_gt_annotations:
                    for pred_idx in range(len(pred_instances_filtered)):
                        pred_inst = pred_instances_filtered[pred_idx]
                        # <--- REFINEMENT: Validar pred_cls_idx
                        pred_cls_idx = pred_inst.pred_classes.item()
                        if not (0 <= pred_cls_idx < num_classes_reference):
                            logger.warning(f"    Pred (img {img_id_str}): ID de clase {pred_cls_idx} fuera de rango [0, {num_classes_reference-1}]. Ignorando esta predicción.")
                            continue
                        pred_score = pred_inst.scores.item()
                        
                        img_y_true.append(background_or_missed_label)
                        img_y_pred.append(pred_cls_idx)
                        fp_bg_per_class[pred_cls_idx] += 1
                        scores_fp_bg.append(pred_score)
                        logger.debug(f"    Pred {pred_idx} (Cls:{pred_cls_idx}, Scr:{pred_score:.2f}) -> FP_bg (No GTs)")
                        if eval_config['analyze_top_k_errors'] > 0:
                            error_analyzer.add_fp(pred_score, file_path, pred_inst.pred_boxes.tensor[0].tolist(), pred_cls_idx, img_id_str)
                else:
                    gt_boxes_list, gt_classes_list = [], []
                    for ann in processed_gt_annotations: # Ya tienen IDs contiguos validados
                        try:
                            box_xyxy = BoxMode.convert(ann["bbox"], ann["bbox_mode"], BoxMode.XYXY_ABS)
                            gt_boxes_list.append(box_xyxy)
                            gt_classes_list.append(ann["category_id"]) 
                        except Exception as e:
                            logger.error(f"Error convirtiendo GT bbox procesado (img {img_id_str}): {e}")
                            continue
                    
                    if not gt_boxes_list: # Si todos los GTs procesados fallaron en conversión de bbox
                        logger.warning(f"    No hay GTs válidos después de conversión de bbox para {img_id_str}. Predicciones serán FP_bg.")
                        for pred_idx in range(len(pred_instances_filtered)):
                            pred_inst = pred_instances_filtered[pred_idx]
                            pred_cls_idx = pred_inst.pred_classes.item()
                            if not (0 <= pred_cls_idx < num_classes_reference): continue # Validar
                            pred_score = pred_inst.scores.item()
                            img_y_true.append(background_or_missed_label); img_y_pred.append(pred_cls_idx)
                            fp_bg_per_class[pred_cls_idx] += 1; scores_fp_bg.append(pred_score)
                            logger.debug(f"    Pred {pred_idx} (Cls:{pred_cls_idx}, Scr:{pred_score:.2f}) -> FP_bg (No GTs válidos post-bbox-conv)")
                            if eval_config['analyze_top_k_errors'] > 0:
                                error_analyzer.add_fp(pred_score, file_path, pred_inst.pred_boxes.tensor[0].tolist(), pred_cls_idx, img_id_str)
                    else:
                        gt_boxes_tensor = Boxes(torch.tensor(np.array(gt_boxes_list), dtype=torch.float32))
                        gt_classes_tensor = torch.tensor(gt_classes_list, dtype=torch.int64) # Ya son contiguos

                        if len(pred_instances_filtered) == 0:
                            for gt_idx, gt_cls_idx_val in enumerate(gt_classes_tensor.tolist()): # gt_cls_idx_val es contiguo
                                img_y_true.append(gt_cls_idx_val)
                                img_y_pred.append(background_or_missed_label)
                                fn_per_class[gt_cls_idx_val] += 1
                                logger.debug(f"    GT {gt_idx} (Cls:{gt_cls_idx_val}) -> FN (No Preds)")
                                if eval_config['analyze_top_k_errors'] > 0:
                                    error_analyzer.add_fn(file_path, gt_boxes_tensor.tensor[gt_idx].tolist(), gt_cls_idx_val, img_id_str)
                        else:
                            pred_boxes_tensor = pred_instances_filtered.pred_boxes
                            pred_classes_tensor = pred_instances_filtered.pred_classes # Ya son contiguos del modelo
                            pred_scores_tensor = pred_instances_filtered.scores
                            
                            ious = pairwise_iou(pred_boxes_tensor, gt_boxes_tensor)
                            iou_thresh_metrics_val = eval_config['iou_thresh_metrics']
                            
                            gt_matched_flags = torch.zeros(gt_boxes_tensor.tensor.shape[0], dtype=torch.bool)
                            pred_matched_to_gt_idx = torch.full((pred_boxes_tensor.tensor.shape[0],), -1, dtype=torch.int64)
                            
                            possible_matches = []
                            for p_idx in range(len(pred_boxes_tensor)):
                                pred_cls_idx_val = pred_classes_tensor[p_idx].item()
                                # <--- REFINEMENT: Validar ID de clase de predicción
                                if not (0 <= pred_cls_idx_val < num_classes_reference):
                                    logger.warning(f"    Pred {p_idx} (img {img_id_str}): ID de clase {pred_cls_idx_val} fuera de rango [0, {num_classes_reference-1}]. Ignorando esta predicción en matching.")
                                    continue
                                for g_idx in range(len(gt_boxes_tensor)):
                                    if ious[p_idx, g_idx] >= iou_thresh_metrics_val:
                                        possible_matches.append((ious[p_idx, g_idx].item(), p_idx, g_idx))
                            possible_matches.sort(key=lambda x: x[0], reverse=True)

                            logger.debug(f"    Found {len(possible_matches)} possible matches (IoU >= {iou_thresh_metrics_val})")
                            for match_idx, (iou_val, p_idx, g_idx) in enumerate(possible_matches):
                                # Pred class ID ya validado arriba
                                pred_cls_actual = pred_classes_tensor[p_idx].item()
                                pred_score_actual = pred_scores_tensor[p_idx].item()
                                gt_cls_actual = gt_classes_tensor[g_idx].item() # Ya validado al crear processed_gt_annotations

                                if not gt_matched_flags[g_idx] and pred_matched_to_gt_idx[p_idx] == -1:
                                    logger.debug(f"      Match Intento {match_idx}: Pred {p_idx} (Cls:{pred_cls_actual}, Scr:{pred_score_actual:.2f}) con GT {g_idx} (Cls:{gt_cls_actual}), IoU:{iou_val:.2f}")
                                    img_y_true.append(gt_cls_actual); img_y_pred.append(pred_cls_actual)
                                    gt_matched_flags[g_idx] = True
                                    pred_matched_to_gt_idx[p_idx] = g_idx
                                    
                                    if gt_cls_actual == pred_cls_actual:
                                        tp_per_class[gt_cls_actual] += 1
                                        scores_tp.append(pred_score_actual)
                                        ious_tp.append(iou_val)
                                        logger.debug(f"        -> TP para clase {gt_cls_actual}")
                                    else: # FP por clasificación incorrecta
                                        fp_cls_per_class[pred_cls_actual] += 1
                                        fn_per_class[gt_cls_actual] += 1 
                                        scores_fp_cls.append(pred_score_actual)
                                        logger.debug(f"        -> FP_Cls para clase predicha {pred_cls_actual} (GT era {gt_cls_actual})")
                                        logger.debug(f"        -> FN para clase GT {gt_cls_actual} (debido a misclassification)")
                                        if eval_config['analyze_top_k_errors'] > 0:
                                            error_analyzer.add_fp(pred_score_actual, file_path, pred_boxes_tensor.tensor[p_idx].tolist(), pred_cls_actual, img_id_str, gt_class_idx_if_misclassified=gt_cls_actual)
                                            # No añadir FN aquí por misclassification, ya se cuenta arriba en fn_per_class[gt_cls_actual]
                                else:
                                    logger.debug(f"      Match Intento {match_idx}: Pred {p_idx} o GT {g_idx} ya matcheado. Saltando.")
                            
                            # FNs (GTs no matcheados por ninguna predicción válida)
                            for g_idx in range(len(gt_boxes_tensor)):
                                if not gt_matched_flags[g_idx]:
                                    gt_cls_actual = gt_classes_tensor[g_idx].item()
                                    img_y_true.append(gt_cls_actual); img_y_pred.append(background_or_missed_label)
                                    fn_per_class[gt_cls_actual] += 1
                                    logger.debug(f"    GT {g_idx} (Cls:{gt_cls_actual}) -> FN (No matcheado)")
                                    if eval_config['analyze_top_k_errors'] > 0:
                                        error_analyzer.add_fn(file_path, gt_boxes_tensor.tensor[g_idx].tolist(), gt_cls_actual, img_id_str)
                            
                            # FPs contra fondo (Predicciones no matcheadas a ningún GT)
                            for p_idx in range(len(pred_boxes_tensor)):
                                pred_cls_actual = pred_classes_tensor[p_idx].item()
                                if not (0 <= pred_cls_actual < num_classes_reference): continue # Ya validado antes
                                
                                if pred_matched_to_gt_idx[p_idx] == -1: # Si la predicción no matcheó con ningún GT
                                    pred_score_actual = pred_scores_tensor[p_idx].item()
                                    img_y_true.append(background_or_missed_label); img_y_pred.append(pred_cls_actual)
                                    fp_bg_per_class[pred_cls_actual] += 1
                                    scores_fp_bg.append(pred_score_actual)
                                    logger.debug(f"    Pred {p_idx} (Cls:{pred_cls_actual}, Scr:{pred_score_actual:.2f}) -> FP_bg (No matcheado a GT)")
                                    if eval_config['analyze_top_k_errors'] > 0:
                                        error_analyzer.add_fp(pred_score_actual, file_path, pred_boxes_tensor.tensor[p_idx].tolist(), pred_cls_actual, img_id_str)
                
                y_true_global.extend(img_y_true)
                y_pred_global.extend(img_y_pred)
    
    # --- FIN DEL BUCLE DE EVALUACIÓN ---

    # ... (Generación de Resultados COCOEvaluator - sin cambios) ...
    logger.info("Generando métricas COCO estándar...")
    coco_eval_summary_str = ""
    coco_results_dict = None 
    try:
        coco_results_dict = coco_evaluator.evaluate() 
        if coco_results_dict:
            logger.info("Resultados de COCOEvaluator (mAP, etc.):")
            coco_results_path = os.path.join(output_dir_abs, "coco_evaluation_results.json")
            with open(coco_results_path, 'w') as f_coco_res: json.dump(coco_results_dict, f_coco_res, indent=4)
            logger.info(f"Resultados detallados de COCOEvaluator guardados en: {coco_results_path}")
            coco_eval_summary_str = "\n\n--- Resultados de COCO Evaluator (Estándar mAP) ---\n"
            for task_name, task_metrics in coco_results_dict.items():
                coco_eval_summary_str += f"Tarea: {task_name}\n"
                for metric_name, value in task_metrics.items():
                    coco_eval_summary_str += f"  {metric_name:<7}: {value:.4f}\n"
        else:
            logger.warning("COCOEvaluator.evaluate() no devolvió resultados.")
            coco_eval_summary_str = "\n\n--- Resultados de COCO Evaluator (Estándar mAP) ---\nNo se generaron resultados.\n"
    except Exception as e:
        logger.error(f"Error durante COCOEvaluator.evaluate(): {e}", exc_info=True)
        coco_eval_summary_str = f"\n\n--- Resultados de COCO Evaluator (Estándar mAP) ---\nError al generar resultados: {e}\n"


    # --- Análisis Avanzado Post-Bucle (usar class_names_reference) ---
    if eval_config['analyze_top_k_errors'] > 0:
        top_fps = error_analyzer.get_top_k_fps()
        top_fns = error_analyzer.get_top_k_fns()
        visualize_errors(top_fps, "false_positives", plots_dir, class_names_reference, eval_config['analyze_top_k_errors'])
        visualize_errors(top_fns, "false_negatives", plots_dir, class_names_reference, eval_config['analyze_top_k_errors'])

    # ... (Plots de distribuciones - sin cambios) ...
    if eval_config['plot_score_distributions']:
        score_dist_data = {"True Positives": scores_tp, "False Positives (Cls)": scores_fp_cls, "False Positives (BG)": scores_fp_bg}
        plot_distributions(score_dist_data, "Distribución de Scores de Detección", "Score", 
                           os.path.join(plots_dir, "score_distributions.png"))
    if eval_config['plot_iou_distribution'] and ious_tp:
        plot_distributions({"True Positives": ious_tp}, "Distribución de IoU para Verdaderos Positivos", "IoU",
                           os.path.join(plots_dir, "iou_distribution_tp.png"), bins=20)


    # --- Métricas Personalizadas y Reporte (usar class_names_reference) ---
    logger.info("Cálculo de métricas personalizadas finales (P/R/F1, Matriz Confusión)...")
    metrics_detailed_path = os.path.join(output_dir_abs, "custom_detailed_counts_per_class.json")
    metrics_detailed = {
        "class_names": class_names_reference, # <--- REFINEMENT
        "tp_per_class": tp_per_class.tolist(),
        "fp_classification_per_class": fp_cls_per_class.tolist(), 
        "fp_background_per_class": fp_bg_per_class.tolist(),   
        "fn_per_class": fn_per_class.tolist(),                 
        "notes": { # (notas sin cambios)
            "fp_classification_per_class": "Predicciones matcheadas a GT pero con clase incorrecta (contado contra clase predicha).",
            "fp_background_per_class": "Predicciones no matcheadas a ningún GT (contado contra clase predicha).",
            "fn_per_class": "GTs no detectados o mal clasificados (contado contra clase real)."
        }
    }
    with open(metrics_detailed_path, 'w') as f: json.dump(metrics_detailed, f, indent=4)
    logger.info(f"Conteos detallados (personalizados) guardados en: {metrics_detailed_path}")

    report_str_custom = ""
    if not y_true_global or not y_pred_global:
        logger.warning("No se generaron datos para métricas personalizadas.")
        report_str_custom = "--- Métricas Personalizadas ---\nNo se generaron datos.\n"
    else:
        # <--- REFINEMENT: usar num_classes_reference y class_names_reference
        labels_for_cm = list(range(num_classes_reference)) + [background_or_missed_label]
        target_names_for_cm = class_names_reference + ["NoDetect/FPbg"]
        cm = confusion_matrix(y_true_global, y_pred_global, labels=labels_for_cm)
        # ... (plot de matriz de confusión - sin cambios en la lógica de plot, pero usa nuevos nombres/labels) ...
        figsize_w = max(10, len(target_names_for_cm) * 0.9); figsize_h = max(8, len(target_names_for_cm) * 0.8)
        plt.figure(figsize=(figsize_w, figsize_h))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names_for_cm, yticklabels=target_names_for_cm, annot_kws={"size": 8})
        plt.xlabel("Clase Predicha"); plt.ylabel("Clase Real")
        plt.title(f"Matriz de Confusión (Métricas Personalizadas, IoU: {eval_config['iou_thresh_metrics']})", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=9); plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout(); cm_path = os.path.join(plots_dir, "confusion_matrix_custom.png")
        plt.savefig(cm_path); plt.close(); logger.info(f"Matriz de confusión (personalizada) guardada en: {cm_path}")

        object_class_labels_custom = list(range(num_classes_reference)) # <--- REFINEMENT
        precision_pc, recall_pc, f1_pc, support_pc = precision_recall_fscore_support(
            y_true_global, y_pred_global, labels=object_class_labels_custom, zero_division=0)
        
        report_str_custom = f"--- Métricas Personalizadas (IoU: {eval_config['iou_thresh_metrics']}, Conf: {eval_config['confidence_threshold']}, EMA: {use_ema_for_inference_flag}) ---\n"
        header = f"{'Clase':<20} | {'Precisión':>10} | {'Recall':>10} | {'F1-score':>10} | {'Support (GTs)':>15}\n"
        separator = "-" * len(header) + "\n"; report_str_custom += separator + header + separator
        
        for i, class_idx in enumerate(object_class_labels_custom):
            name = class_names_reference[class_idx] # <--- REFINEMENT
            report_str_custom += (f"{name:<20} | {precision_pc[i]:10.4f} | {recall_pc[i]:10.4f} | "
                                  f"{f1_pc[i]:10.4f} | {support_pc[i]:15}\n")
        # ... (resto del reporte de P/R/F1 y notas como antes) ...
        report_str_custom += separator
        precision_macro = np.mean(precision_pc); recall_macro = np.mean(recall_pc); f1_macro = np.mean(f1_pc)
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(y_true_global, y_pred_global, labels=object_class_labels_custom, average='weighted', zero_division=0)
        total_support = np.sum(support_pc)
        report_str_custom += f"{'Promedio Macro':<20} | {precision_macro:10.4f} | {recall_macro:10.4f} | {f1_macro:10.4f} | {'-':>15}\n"
        report_str_custom += f"{'Promedio Ponderado':<20} | {precision_w:10.4f} | {recall_w:10.4f} | {f1_w:10.4f} | {total_support:15}\n"
        report_str_custom += separator
        report_str_custom += "Notas (Métricas Personalizadas):\n - ... (ver v3) ...\n"


    # ... (Reporte de Tiempo de Inferencia y Configuración como antes) ...
    avg_time_per_batch = total_inference_time / num_batches_processed if num_batches_processed > 0 else 0
    avg_time_per_image = total_inference_time / num_images_processed if num_images_processed > 0 else 0
    inference_time_report = "\n\n--- Análisis de Tiempo de Inferencia ---\n" # ... (como antes)
    inference_time_report += f"Imágenes procesadas: {num_images_processed}\n"
    inference_time_report += f"Batches procesados: {num_batches_processed}\n"
    inference_time_report += f"Tiempo total de inferencia: {total_inference_time:.2f} segundos\n"
    inference_time_report += f"Tiempo promedio por batch: {avg_time_per_batch:.4f} segundos\n"
    inference_time_report += f"Tiempo promedio por imagen: {avg_time_per_image:.4f} segundos\n"

    config_summary_report = "\n\n--- Resumen de Configuración de Evaluación ---\n" # ... (como antes)
    config_summary_report += f"Archivo de config. del modelo: {eval_config['model_config_file']}\n"
    config_summary_report += f"Pesos del modelo: {eval_config['model_weights']}\n"
    config_summary_report += f"Dataset: {eval_config['dataset_name']} ({eval_config['annotation_file']})\n"
    config_summary_report += f"Clases de referencia (Metadata): {class_names_reference}\n" # <--- REFINEMENT
    config_summary_report += f"Directorio de salida: {output_dir_abs}\n" # ... (resto como antes)
    config_summary_report += f"Usar pesos EMA: {eval_config['use_ema_weights']} (Efectivo: {use_ema_for_inference_flag})\n"
    config_summary_report += f"Umbral de confianza (custom): {eval_config['confidence_threshold']}\n"
    config_summary_report += f"Umbral IoU (custom): {eval_config['iou_thresh_metrics']}\n"
    config_summary_report += f"Analizar Top-K errores: {eval_config['analyze_top_k_errors']}\n"
    config_summary_report += f"Plotear distribuciones de score: {eval_config['plot_score_distributions']}\n"
    config_summary_report += f"Plotear distribución de IoU: {eval_config['plot_iou_distribution']}\n"
    config_summary_report += f"Detectron2 CFG - INPUT.MIN_SIZE_TEST: {d2_cfg.INPUT.MIN_SIZE_TEST}\n"
    config_summary_report += f"Detectron2 CFG - INPUT.MAX_SIZE_TEST: {d2_cfg.INPUT.MAX_SIZE_TEST}\n"
    if hasattr(d2_cfg.MODEL, 'ROI_HEADS') and hasattr(d2_cfg.MODEL.ROI_HEADS, 'SCORE_THRESH_TEST'):
         config_summary_report += f"Detectron2 CFG - ROI_HEADS.SCORE_THRESH_TEST: {d2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}\n"


    final_report_str = config_summary_report + \
                       report_str_custom + \
                       coco_eval_summary_str + \
                       inference_time_report
    
    logger.info("\n" + final_report_str)
    report_path = os.path.join(output_dir_abs, "evaluation_report_FULL.txt")
    with open(report_path, "w") as f: f.write(final_report_str)
    logger.info(f"Reporte COMPLETO guardado en: {report_path}")
    logger.info(f"Visualizaciones de inferencia estándar guardadas en: {inference_images_dir}")
    logger.info(f"Plots y análisis avanzados (errores, distribuciones) guardados en: {plots_dir}")

def main():
    # (Parser de argumentos y config de logging como antes)
    parser = argparse.ArgumentParser(description="Script de evaluación AVANZADO para DiffusionDet.")
    # ... (argumentos como antes)
    parser.add_argument('--config', type=str, default='evaluation_config.yaml', help="Ruta al archivo YAML de configuración de la evaluación.")
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Nivel de logging.")
    args = parser.parse_args()
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int): raise ValueError(f'Nivel de log inválido: {args.log_level}')
    logging.basicConfig(level=numeric_level, format='%(asctime)s %(levelname)s [%(name)s][%(funcName)s]: %(message)s') # Añadido funcName

    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_config_path = args.config
    if not os.path.isabs(eval_config_path): eval_config_path = os.path.join(script_dir, eval_config_path)
    
    eval_config = load_evaluation_config(eval_config_path)
    d2_cfg = setup_detectron2_cfg(eval_config)
    register_evaluation_dataset(eval_config) 
    
    # ... (Verificaciones de metadata como antes) ...
    metadata_check = MetadataCatalog.get(eval_config['dataset_name'])
    if not metadata_check.get("json_file"):
        sys.exit(f"¡Error Crítico! Metadata para '{eval_config['dataset_name']}' no tiene 'json_file'.")
    if not metadata_check.thing_classes: # Ya se verifica y sale en register_evaluation_dataset si es necesario
        sys.exit("¡Error Crítico! No se pudieron determinar las clases del dataset desde metadata.")

    evaluate_model_aligned(d2_cfg, eval_config)
    logger.info("Proceso de evaluación AVANZADO finalizado.")

if __name__ == "__main__":
    main()
