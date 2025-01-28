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

import torch
import numpy as np
from fvcore.nn.precise_bn import get_bn_modules

from detectron2.data import build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import Instances, BoxMode

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import TensorboardXWriter
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, create_ddp_model, \
    AMPTrainer, SimpleTrainer, hooks
from detectron2.evaluation import COCOEvaluator, LVISEvaluator, verify_results
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.modeling import build_model

from diffusiondet import DiffusionDetDatasetMapper, add_diffusiondet_config, DiffusionDetWithTTA
from diffusiondet.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer

logger = logging.getLogger(__name__)

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
        self.img_format = cfg.INPUT.FORMAT  # p.e. "RGB"
        
        # Definimos varias transformaciones
        if self.is_train:
            self.tfm_gens = [
                # 1) Rotación aleatoria en un rango ±15°
                T.RandomRotation(
                    angle=[-15, 15],
                    expand=False,     # no expandir la imagen
                    center=None,
                    sample_style="range"
                ),
                # 2) Flip horizontal (ya está en config, pero lo forzamos aquí también)
                T.RandomFlip(horizontal=True, vertical=False),
                # 3) Ajuste de brillo (escala 0.8 a 1.2)
                T.RandomBrightness(0.8, 1.2),
                # 4) Ajuste de contraste
                T.RandomContrast(0.8, 1.2),
                # 5) Ajuste de saturación (si las imágenes no son monocromáticas)
                T.RandomSaturation(0.8, 1.2),
                # 6) Ajuste ligero de color (simula distintas condiciones de luz)
                T.RandomLighting(0.7),
                # Nota: Si quieres recorte aleatorio, puedes añadir T.RandomCrop(...)
            ]
        else:
            # Para validación/test, NO se aplican augmentations
            self.tfm_gens = []

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
        transforms = T.AugmentationList(self.tfm_gens)(aug_input)
        image = aug_input.image  # imagen transformada

        # 3. Procesar anotaciones SOLO si estamos en entrenamiento y hay "annotations"
        if self.is_train and "annotations" in dataset_dict:
            annos = dataset_dict["annotations"]
            valid_annos = []
            for obj in annos:
                # Asegurarnos de que sea formato COCO [x, y, w, h]
                if "bbox" not in obj:
                    continue
                coco_bbox = obj["bbox"]
                if len(coco_bbox) != 4:
                    continue

                # Convertir [x, y, w, h] -> [x_min, y_min, x_max, y_max]
                x0, y0, w, h = coco_bbox
                x1 = x0 + w
                y1 = y0 + h

                # Clave: Detectron2 espera BOX en formato XYXY_ABS
                obj["bbox"] = [x0, y0, x1, y1]
                obj["bbox_mode"] = BoxMode.XYXY_ABS

                # Filtramos cajas con w<=0 o h<=0
                if w <= 0 or h <= 0:
                    logger.warning(
                        f"Descartando bbox con w<=0 o h<=0: {obj['bbox']} en imagen {dataset_dict['file_name']}"
                    )
                    continue

                # Agregar a lista las anotaciones "sanas"
                valid_annos.append(obj)

            # Reemplazamos con las anotaciones filtradas
            dataset_dict["annotations"] = valid_annos

            # 4. Aplicar transformaciones a anotaciones
            #    transform_instance_annotations convierte XYXY según "transforms"
            for obj in dataset_dict["annotations"]:
                # Aplica la misma aug a la caja
                try:
                    utils.transform_instance_annotations(
                        annotation=obj, transforms=transforms, image_size=image.shape[:2]
                    )
                except Exception as e:
                    # Si algo falla (caja fuera, etc.), descarta la anotación
                    logger.warning(
                        f"Ocurrió un error transformando anotación {obj['bbox']} -> se descarta. Error: {e}"
                    )
                    obj["ignore"] = True  # Marcamos para filtrar luego

            # Filtrar las anotaciones marcadas como 'ignore' o con boxes inválidos
            dataset_dict["annotations"] = [
                anno
                for anno in dataset_dict["annotations"]
                if not anno.get("ignore", False)
            ]

            # 5. Revisar cajas resultantes, descartar/bloquear NaNs o x_max < x_min
            final_annos = []
            for anno in dataset_dict["annotations"]:
                bb = anno["bbox"]  # [xmin, ymin, xmax, ymax]
                if any(np.isnan(bb_i) for bb_i in bb):
                    logger.warning(
                        f"Se encontró NaN en la bbox {bb}. Se descarta la anotación."
                    )
                    continue

                x0, y0, x1, y1 = bb
                if x1 < x0 or y1 < y0:
                    logger.warning(
                        f"Coordenadas invertidas {bb}. Se descarta la anotación."
                    )
                    continue

                # Mantenerla si es válida
                final_annos.append(anno)

            dataset_dict["annotations"] = final_annos

            # 6. Convertir anotaciones a objeto Instances
            #    (necesario para que el modelo reciba "instances")
            if len(dataset_dict["annotations"]) == 0:
                # Si no quedan anotaciones, metemos un Instances vacío
                dataset_dict["instances"] = Instances(image.shape[:2])
            else:
                # Caso normal: creamos las instancias
                dataset_dict["instances"] = utils.annotations_to_instances(
                    dataset_dict["annotations"], image.shape[:2]
                )
        else:
            # Modo test/val o sin anotaciones
            dataset_dict["instances"] = Instances(image.shape[:2])

        # 7. Convertir la imagen a tensor CHW
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        return dataset_dict


# ---------------------------------------------------------------------------- #
# 2) DEFINICIÓN DEL TRAINER ESPECIALIZADO
# ---------------------------------------------------------------------------- #
class Trainer(DefaultTrainer):
    """
    Adaptación de DefaultTrainer para DiffusionDet y usando un mapper de dataset personalizado.
    """

    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()

        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        # Crear modelo en DDP si corresponde
        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # EMA Setup
        kwargs = {
            'trainer': weakref.proxy(self),
        }
        kwargs.update(may_get_ema_checkpointer(cfg, model))
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

        # Save the configuration for reproducibility
        self._save_config()

    def _save_config(self):
        """
        Guarda la configuración (cfg.dump()) en OUTPUT_DIR
        """
        config_path = os.path.join(self.cfg.OUTPUT_DIR, 'config.yaml')
        with open(config_path, 'w') as f:
            f.write(self.cfg.dump())
        logging.getLogger(__name__).info(f"Configuration saved to {config_path}")
        
    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        may_build_model_ema(cfg, model)
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Devuelve evaluador de tipo COCO o LVIS según dataset_name.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        os.makedirs(output_folder, exist_ok=True)
        if 'lvis' in dataset_name:
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        else:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Sobrescribimos este método para usar nuestro MarineDebrisDatasetMapper.
        """
        mapper = MarineDebrisDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

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
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
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
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = DiffusionDetWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA", name)
            )
            for name in cfg.DATASETS.TEST
        ]
        if cfg.MODEL_EMA.ENABLED:
            cls.ema_test(cfg, model, evaluators)
        else:
            res = cls.test(cfg, model, evaluators)
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

        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)

            # Save metrics to a JSON file for later analysis
            metrics_path = os.path.join(cfg.OUTPUT_DIR, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(self._last_eval_results, f)
            logging.getLogger(__name__).info(f"Metrics saved to {metrics_path}")

            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

# ---------------------------------------------------------------------------- #
# 3) SETUP Y MAIN DE ENTRENAMIENTO
# ---------------------------------------------------------------------------- #
def setup(args):
    """
    Cargar configs, fusionar con args, e inicializar.
    """
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    # Load custom datasets
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import MetadataCatalog

    # ------------------------- #
    # REGISTRO DE DATASETS
    # ------------------------- #
    data_root = '/app/DiffusionDet/Densea/datasets'  # Adjust as needed

    register_coco_instances("cleansea_train", {}, os.path.join(data_root, "cleansea/train/annotations_bbox.json"), os.path.join(data_root, "cleansea/train"))
    register_coco_instances("cleansea_train_grouped", {}, os.path.join(data_root, "cleansea/train/annotations_bbox_grouped.json"), os.path.join(data_root, "cleansea/train"))
    register_coco_instances("cleansea_test", {}, os.path.join(data_root, "cleansea/test/annotations_bbox.json"), os.path.join(data_root, "cleansea/test"))
    register_coco_instances("cleansea_test_grouped", {}, os.path.join(data_root, "cleansea/test/annotations_bbox_grouped.json"), os.path.join(data_root, "cleansea/test"))
    register_coco_instances("synthetic_v1", {}, os.path.join(data_root, "synthetic/v1/annotations_bbox.json"), os.path.join(data_root, "synthetic/v1"))
    register_coco_instances("synthetic_v1_grouped", {}, os.path.join(data_root, "synthetic/v1/annotations_bbox_grouped.json"), os.path.join(data_root, "synthetic/v1"))
    register_coco_instances("synthetic_v2_train", {}, os.path.join(data_root, "synthetic/v2/train_coco/annotations_bbox.json"), os.path.join(data_root, "synthetic/v2/train_coco"))
    register_coco_instances("synthetic_v2_test", {}, os.path.join(data_root, "synthetic/v2/test_coco/annotations_bbox.json"), os.path.join(data_root, "synthetic/v2/test_coco"))

    # Definimos las clases
    classes = ["background","Can","Squared_Can","Wood","Bottle","Plastic_Bag","Glove","Fishing_Net","Tire","Packaging_Bag",
               "WashingMachine","Metal_Chain","Rope","Towel","Plastic_Debris","Metal_Debris","Pipe","Shoe",
               "Car_Bumper","Basket"]

    grouped_classes = ['Small_Lightweight_Debris', 'Wooden_Debris', 'Textile_Debris', 'Flexible_Plastic_Debris', 'Large_Heavy_Debris', 'Small_Metal_Debris']

    # Update Metadata
    datasets = ["cleansea_train", "cleansea_train_grouped", "cleansea_test", "cleansea_test_grouped", "synthetic_v1", "synthetic_v1_grouped", "synthetic_v2_train", "synthetic_v2_test"]
    for dataset in datasets:
        MetadataCatalog.get(dataset).set(thing_classes=classes)

    # Save configuration
    with open(os.path.join(cfg.OUTPUT_DIR, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())

    if args.eval_only:
        model = Trainer.build_model(cfg)
        kwargs = may_get_ema_checkpointer(cfg, model)
        if cfg.MODEL_EMA.ENABLED:
            EMADetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(cfg.MODEL.WEIGHTS,
                                                                                              resume=args.resume)
        else:
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(cfg.MODEL.WEIGHTS,
                                                                                           resume=args.resume)
        res = Trainer.ema_test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

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
