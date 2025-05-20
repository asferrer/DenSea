#!/usr/bin/env python
"""
DiffusionDet Training Script.
"""

import os
import copy
import itertools
import weakref
from typing import Any, Dict, List, Set
import logging
from collections import OrderedDict
import json
from PIL import ImageFilter
import cv2

import torch
import numpy as np
import time

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import get_event_storage
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from fvcore.nn.precise_bn import get_bn_modules

from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import Instances, BoxMode
from detectron2.engine import (
    DefaultTrainer, default_argument_parser, default_setup, launch,
    create_ddp_model, AMPTrainer, SimpleTrainer, hooks
)
from detectron2.evaluation import COCOEvaluator, LVISEvaluator, verify_results, DatasetEvaluators
from detectron2.solver.build import maybe_add_gradient_clipping, build_lr_scheduler
from detectron2.modeling import build_model
from torch.amp import autocast as cuda_autocast, GradScaler

from diffusiondet import (
    add_diffusiondet_config,
    DiffusionDetWithTTA
)

from diffusiondet.util.model_ema import (
    add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer,
    EMAHook, apply_model_ema_and_restore, EMADetectionCheckpointer
)

logger = logging.getLogger("detectron2")
logger.setLevel(logging.INFO)
class _GaussianBlurTransform(T.Transform):
    def __init__(self, radius: float): super().__init__(); self.radius = radius
    def apply_image(self, img: np.ndarray) -> np.ndarray: return cv2.GaussianBlur(img, ksize=(0,0), sigmaX=self.radius, sigmaY=self.radius)
    def apply_coords(self, coords: np.ndarray) -> np.ndarray: return coords

class RandomGaussianBlur(T.Augmentation):
    def __init__(self, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0): super().__init__(); self._init(locals())
    def get_transform(self, image: np.ndarray) -> T.Transform:
        if torch.rand(1) < self.p:
            radius = torch.empty(1).uniform_(self.radius_min, self.radius_max).item()
            return _GaussianBlurTransform(radius)
        return T.NoOpTransform()

class _NoiseTransform(T.Transform):
    def __init__(self, std_dev_abs: float): super().__init__(); self.std_dev_abs = std_dev_abs
    def apply_image(self, img: np.ndarray) -> np.ndarray:
        img_float = img.astype(np.float32)
        noise = np.random.normal(0, self.std_dev_abs, img_float.shape)
        return np.clip(img_float + noise, 0, 255).astype(np.uint8)
    def apply_coords(self, coords: np.ndarray) -> np.ndarray: return coords

class RandomNoise(T.Augmentation):
    def __init__(self, p: float = 0.5, noise_std_dev_max: float = 0.03): super().__init__(); self._init(locals())
    def get_transform(self, image: np.ndarray) -> T.Transform:
        if torch.rand(1) < self.p:
            std_dev_abs = torch.empty(1).uniform_(0, self.noise_std_dev_max).item() * 255.0
            return _NoiseTransform(std_dev_abs)
        return T.NoOpTransform()
    
# ---------------------------------------------------------------------------- #
# 1) DEFINICIÓN DE UN DATASET MAPPER PERSONALIZADO PARA DATA AUGMENTATION
# ---------------------------------------------------------------------------- #
class MarineDebrisDatasetMapper:
    """
    Mapper personalizado para inyectar data augmentation extra,
    manteniendo compatibilidad con la detección de objetos.
    """

    def __init__(self, cfg, is_train=True):
        """
        Args:
            cfg: configuración detectron2
            is_train (bool): true para entrenamiento
        """
        self.is_train = is_train
        self.img_format = cfg.INPUT.FORMAT
        augmentations = []
        # Definimos varias transformaciones
        if self.is_train:
            augmentations.extend([
                T.RandomFlip(horizontal=True, vertical=False), T.RandomBrightness(0.6, 1.4), 
                T.RandomContrast(0.6, 1.4), T.RandomSaturation(0.6, 1.4),
                T.RandomRotation(angle=[-20, 20], expand=False, sample_style="range"),
                RandomGaussianBlur(p=0.4, radius_min=0.5, radius_max=2.5), 
                RandomNoise(p=0.3, noise_std_dev_max=0.03), 
            ])
        self.tfm_gens = T.AugmentationList(augmentations)
        logger.info(f"MarineDebrisDatasetMapper (is_train={is_train}) using augmentations: {self.tfm_gens}")

    def __call__(self, dataset_dict):
        """
        dataset_dict: diccionario con:
          - "file_name": ruta de la imagen
          - "annotations": lista de anotaciones (opcional, si is_train)
          - ... (otros metadatos)
        Retorna: dict modificado con:
          - "image": tensor [C, H, W]
          - "instances": objeto Instances con gt_boxes, gt_classes...
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # Evita mutar el original

        # 1. Cargar la imagen
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # 2. Aplicar transformaciones de augmentation
        aug_input = T.AugInput(image)
        transforms_applied = self.tfm_gens(aug_input)
        image_transformed = aug_input.image
        dataset_dict["image_shape"] = image_transformed.shape

        if "annotations" in dataset_dict:
            annos = []
            for obj in dataset_dict.get("annotations", []):
                if obj.get("iscrowd", 0) == 1: continue
                if "bbox" not in obj: continue
                bbox_mode = obj.get("bbox_mode", BoxMode.XYWH_ABS)
                bbox_xyxy = BoxMode.convert(obj["bbox"], bbox_mode, BoxMode.XYXY_ABS)
                instance_dict_to_transform = {"bbox": bbox_xyxy, "bbox_mode": BoxMode.XYXY_ABS}
                if "segmentation" in obj: instance_dict_to_transform["segmentation"] = obj["segmentation"]
                transformed_instance = utils.transform_instance_annotations(instance_dict_to_transform, transforms_applied, image_transformed.shape[:2])
                final_bbox = transformed_instance["bbox"]
                if final_bbox[2] > final_bbox[0] and final_bbox[3] > final_bbox[1]:
                    obj["bbox"] = final_bbox; obj["bbox_mode"] = transformed_instance["bbox_mode"]
                    if "segmentation" in transformed_instance: obj["segmentation"] = transformed_instance["segmentation"]
                    annos.append(obj)
            dataset_dict["annotations"] = annos
            instances = utils.annotations_to_instances(annos, image_transformed.shape[:2])
            if self.is_train: dataset_dict["instances"] = utils.filter_empty_instances(instances)
            else: dataset_dict["instances"] = instances
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image_transformed.transpose(2, 0, 1)))
        return dataset_dict

# ---------------------------------------------------------------------------- #
# 2) DEFINICIÓN DEL TRAINER ESPECIALIZADO
# ---------------------------------------------------------------------------- #
class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        logger_trainer = logging.getLogger("detectron2")
        if not logger_trainer.isEnabledFor(logging.INFO): setup_logger()
        
        cfg_init = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        
        self.model = self.build_model(cfg_init)
        self.optimizer = self.build_optimizer(cfg_init, self.model)
        self.data_loader = self.build_train_loader(cfg_init)
        self._data_loader_iter = iter(self.data_loader)

        self.model = create_ddp_model(self.model, broadcast_buffers=False, 
                                      find_unused_parameters=cfg_init.MODEL.BACKBONE.FREEZE_AT > -1 or cfg_init.get("FIND_UNUSED_PARAMETERS", False))
        self.accumulation_steps = 1
        if self.accumulation_steps > 1:
            logger_trainer.info(f"Gradient accumulation enabled: {self.accumulation_steps} steps.")
            self.optimizer.zero_grad() 

        if cfg_init.SOLVER.AMP.ENABLED: self.grad_scaler = GradScaler()
        else: self.grad_scaler = None
        
        self.scheduler = Trainer.build_lr_scheduler(cfg_init, self.optimizer)

        kwargs_ema_checkpointer = {'trainer': weakref.proxy(self)}
        kwargs_ema_checkpointer.update(may_get_ema_checkpointer(cfg_init, self.model))

        self.checkpointer = DetectionCheckpointer(
            self.model, cfg_init.OUTPUT_DIR,
            optimizer=self.optimizer, scheduler=self.scheduler,
            **kwargs_ema_checkpointer)
        
        self.start_iter = 0 
        self.max_iter = cfg_init.SOLVER.MAX_ITER
        self.cfg = cfg_init
        
        self.register_hooks(self.build_hooks())
        self._save_config()

    def _save_config(self): # Idéntico
        config_path = os.path.join(self.cfg.OUTPUT_DIR, 'config_runtime.yaml')
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        with open(config_path, 'w') as f: f.write(self.cfg.dump())
        logging.getLogger(__name__).info(f"Runtime configuration saved to {config_path}")

    def run_step(self): 
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        start = time.perf_counter()
        try: data = next(self._data_loader_iter)
        except StopIteration: self._data_loader_iter = iter(self.data_loader); data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start
        
        loss_scaler_for_accum = 1.0 / self.accumulation_steps if self.accumulation_steps > 1 else 1.0

        with cuda_autocast(enabled=self.cfg.SOLVER.AMP.ENABLED, dtype=torch.float16 if self.cfg.SOLVER.AMP.get("PRECISION", "fp16") == "fp16" else torch.bfloat16, device_type="cuda"):
            loss_dict = self.model(data) 
            losses = sum(loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict
        
        if torch.isnan(losses) or torch.isinf(losses):
            logger.error(f"NaN/Inf loss detected at iter {self.iter} BEFORE backward: {loss_dict}")
            # Opcional: guardar datos de entrada para depuración
            # torch.save(data, f"debug_data_iter_{self.iter}.pt")
            raise FloatingPointError(f"Loss became NaN/Inf at iter {self.iter} before backward pass. loss_dict: {loss_dict}")

        if self.cfg.SOLVER.AMP.ENABLED:
            self.grad_scaler.scale(losses * loss_scaler_for_accum).backward()
        else:
            (losses * loss_scaler_for_accum).backward()

        if (self.iter + 1) % self.accumulation_steps == 0:
            if self.cfg.SOLVER.AMP.ENABLED:
                self.grad_scaler.unscale_(self.optimizer) 
                if self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                if self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE)
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        storage = get_event_storage()
        # Registrar cada componente de la pérdida y la pérdida total
        if isinstance(loss_dict, dict):
            # Asegurar que los tensores estén en CPU y sean float para el storage
            metrics_to_log = {k: v.detach().cpu().item() for k,v in loss_dict.items()}
            storage.put_scalars(**metrics_to_log) # Pasa cada clave-valor como kwarg
            # También registrar la pérdida total si no está ya en loss_dict
            if "total_loss" not in metrics_to_log : # En caso de que el modelo no devuelva 'total_loss'
                 storage.put_scalar("total_loss", losses.detach().cpu().item())
        else: # Si el modelo solo devuelve la pérdida total
            storage.put_scalar("total_loss", losses.detach().cpu().item())
            
        storage.put_scalar("data_time", data_time)
    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        # Configura EMA si está habilitado en la config
        if cfg.MODEL_EMA.ENABLED: may_build_model_ema(cfg, model)
        return model
    
    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """Construye el planificador de tasa de aprendizaje desde la config."""
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Devuelve evaluador de tipo COCO o LVIS según dataset_name.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        os.makedirs(output_folder, exist_ok=True)
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "lvis":
            evaluator_list.append(LVISEvaluator(dataset_name, cfg, True, output_folder))
        elif evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                f"No evaluator found for dataset {dataset_name} with type {evaluator_type}"
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Sobrescribimos este método para usar nuestro MarineDebrisDatasetMapper.
        """
        mapper = MarineDebrisDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Sobrescribe para usar nuestro EnhancedMarineDebrisDatasetMapper en modo test.
        """
        mapper = MarineDebrisDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Construye el optimizador (SGD o AdamW), con soporte a gradient clipping.
        """
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"No optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def ema_test(cls, cfg, model, evaluators=None):
        """
        Si se habilitó EMA, se evalúa con EMA, sino se evalúa normal.
        """
        logger = logging.getLogger("detectron2.trainer")
        if cfg.MODEL_EMA.ENABLED:
            logger.info("Run evaluation with EMA.")
            with apply_model_ema_and_restore(model):
                results = cls.test(cfg, model, evaluators=evaluators)
        else:
            results = cls.test(cfg, model, evaluators=evaluators)
        return results

    @classmethod
    def test_with_TTA(cls, cfg, model):
        """Ejecuta inferencia con Test-Time Augmentation."""
        logger.info("Running inference with test-time augmentation (TTA)...")
        # Construir evaluadores específicos para TTA si es necesario (directorios diferentes)
        evaluators = []
        for name in cfg.DATASETS.TEST:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_TTA", name)
            evaluator = cls.build_evaluator(cfg, name, output_folder=output_folder)
            evaluators.append(evaluator)
        evaluators = DatasetEvaluators(evaluators) # Combinar si hay múltiples

        # Envolver modelo con TTA y ejecutar evaluación (con o sin EMA)
        tta_model = DiffusionDetWithTTA(cfg, model)
        if cfg.MODEL_EMA.ENABLED:
            res = cls.ema_test(cfg, tta_model, evaluators=evaluators)
        else:
            res = cls.test(cfg, tta_model, evaluators=evaluators)

        # Añadir sufijo _TTA a las claves de resultados
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def build_hooks(self):
        """
        Lista de hooks (checkpoint, eval, logger, etc.).
        """
        cfg = self.cfg.clone()
        cfg.defrost()

        ret = [
            hooks.IterationTimer(),
            EMAHook(self.cfg, self.model) if cfg.MODEL_EMA.ENABLED else None,
            hooks.LRScheduler(),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]
        # Eliminar None de la lista si EMA o PreciseBN no se añadieron
        ret = [hook for hook in ret if hook is not None]

        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(
                self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_to_keep=5 # Guardar últimos 5 checkpoints
            ))
        def test_and_save_results():
            # Evaluar usando EMA si está habilitado
            self._last_eval_results = self.ema_test(self.cfg, self.model)

            # Guardar métricas en un JSON (solo proceso principal)
            if comm.is_main_process():
                 metrics_path = os.path.join(cfg.OUTPUT_DIR, 'metrics.json')
                 try:
                     with open(metrics_path, 'w') as f:
                         # Convertir tipos numpy a nativos de Python para JSON
                         serializable_results = convert_dict_values_to_serializable(self._last_eval_results)
                         json.dump(serializable_results, f, indent=4)
                     logger.info(f"Metrics saved to {metrics_path}")
                 except Exception as e:
                     logger.error(f"Failed to save metrics to {metrics_path}: {e}")

            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

# Función auxiliar para convertir valores numpy a tipos nativos para JSON
def convert_dict_values_to_serializable(d):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            new_dict[k] = convert_dict_values_to_serializable(v)
        elif isinstance(v, (np.float32, np.float64)):
            new_dict[k] = float(v)
        elif isinstance(v, (np.int32, np.int64)):
            new_dict[k] = int(v)
        elif isinstance(v, list):
             # Convertir elementos de la lista si son numpy types
             new_dict[k] = [float(i) if isinstance(i, (np.float32, np.float64)) else
                            int(i) if isinstance(i, (np.int32, np.int64)) else i
                            for i in v]
        else:
            new_dict[k] = v
    return new_dict

# ---------------------------------------------------------------------------- #
# 3) SETUP Y MAIN DE ENTRENAMIENTO
# ---------------------------------------------------------------------------- #
def setup(args):
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Crear directorio de salida
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    default_setup(cfg, args)
    # Configura el logger principal después de crear OUTPUT_DIR
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="detectron2")

    return cfg

def register_datasets(data_root):
    """Registra todos los datasets necesarios en formato COCO."""
    logger.info(f"Registrando datasets desde la raíz: {data_root}")

    # --- Metadatos Comunes ---
    # Asegúrate que las clases aquí coinciden exactamente con las de tus archivos JSON
    densea_metadata = {
        # "thing_classes": [
        #     "Basket", "Bottle", "Can", "Car_Bumper", "Fishing_Net",
        #     "Glove", "Mask", "Metal_Chain", "Metal_Debris", "Packaging_Bag",
        #     "Pipe", "Plastic_Bag", "Plastic_Debris", "Rope", "Shoe",
        #     "Squared_Can", "Tire", "Towel", "WashingMachine", "Wood"
        # ]
        "thing_classes": [
            "Bottle", "Can", "Fishing_Net",
            "Glove", "Mask", "Metal_Debris",
            "Plastic_Debris", "Tire"
        ]
    }
    num_classes = len(densea_metadata["thing_classes"])
    logger.info(f"Número de clases definidas: {num_classes}")

    datasets_to_register = {
        "densea_train": ("split_v7_diffusiondet/train.json", "datasets"),
        "densea_valid": ("split_v7_diffusiondet/val.json", "datasets"),
        "densea_test": ("split_v7_diffusiondet/test.json", "datasets"),
    }

    for name, (json_file, image_root) in datasets_to_register.items():
        json_path = os.path.join(data_root, json_file)
        image_path = image_root
        # Verificar si los archivos/directorios existen antes de registrar
        if os.path.exists(json_path) and os.path.isdir(image_path):
            logger.info(f"Registrando: {name}")
            register_coco_instances(name, densea_metadata, json_path, image_path)
            # Establecer evaluator_type para cada dataset registrado
            MetadataCatalog.get(name).evaluator_type = "coco" # o "lvis" si corresponde
            try:
                data = DatasetCatalog.get(name)
                logger.info(f"Dataset '{name}' registrado con {len(data)} imágenes.")
            except Exception as e:
                logger.error(f"Error al verificar el registro de '{name}': {e}")
        else:
            logger.warning(f"No se pudo registrar '{name}': JSON ('{json_path}') o Directorio de Imágenes ('{image_path}') no encontrado.")

    logger.info("Registro de datasets completado.")
    return num_classes

def main(args):
    cfg = setup(args)

    # --- Registrar Datasets ---
    data_root = '/app/DiffusionDet/Densea' # Valor por defecto si no está en config
    logger.warning(f"DATASETS.ROOT no definido en config, usando por defecto: {data_root}")
    num_classes = register_datasets(data_root)
    cfg.defrost()
    model_cfg_updated = False

    # --- Actualizar número de clases en la configuración ---
    # Ejemplo genérico (ajusta según tu config base):
    if hasattr(cfg.MODEL, 'ROI_HEADS'): # Para arquitecturas tipo Faster R-CNN
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        model_cfg_updated = True
    if hasattr(cfg.MODEL, 'DiffusionDet') and hasattr(cfg.MODEL.DiffusionDet, 'NUM_CLASSES'):
         cfg.MODEL.DiffusionDet.NUM_CLASSES = num_classes
         model_cfg_updated = True
    elif hasattr(cfg.MODEL, 'num_classes'): # A veces está directamente en MODEL
         cfg.MODEL.num_classes = num_classes
         model_cfg_updated = True
    
    if not model_cfg_updated: logger.warning("No se encontró un campo NUM_CLASSES estándar en cfg.MODEL.")
    else: logger.info(f"Configuración actualizada con NUM_CLASSES = {num_classes}")
    
    # Congelar la configuración después de los ajustes finales
    cfg.freeze()

    # Volver a guardar la config actualizada (opcional, útil para debug)
    if comm.is_main_process():
        config_path = os.path.join(cfg.OUTPUT_DIR, 'config_final_updated.yaml')
        with open(config_path, 'w') as f: f.write(cfg.dump())

    # Modo evaluación solamente
    if args.eval_only:
        model = Trainer.build_model(cfg)
        kwargs = may_get_ema_checkpointer(cfg, model)
        if cfg.MODEL_EMA.ENABLED:
            EMADetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
        else:
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
        res = Trainer.ema_test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    # Entrenamiento normal
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )