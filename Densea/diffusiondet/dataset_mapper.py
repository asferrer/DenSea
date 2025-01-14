import os
import copy
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2  # Importar cv2 para usar en PadIfNeeded

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog
import albumentations as A
from albumentations.pytorch import ToTensorV2

__all__ = ["DiffusionDetDatasetMapper"]  # Asegúrate de que este nombre coincida con tu clase


class DiffusionDetDatasetMapper:
    """
    Custom Dataset Mapper that applies class grouping, data augmentation suitable for underwater images,
    handles both original and grouped classes, and monitors class distribution before and after augmentation.
    """

    def __init__(self, cfg, is_train=True, use_grouped_classes=True, output_dir="./output"):
        self.is_train = is_train
        self.use_grouped_classes = use_grouped_classes
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.image_format = cfg.INPUT.FORMAT
        # Reconstruir el diccionario de agrupamiento de clases
        self.class_grouping = dict(cfg.CLASS_GROUPING)
        self.original_class_names = cfg.ORIGINAL_CLASSES

        # Determinar el nombre actual del dataset
        if is_train:
            dataset_name = cfg.DATASETS.TRAIN[0]
        else:
            dataset_name = cfg.DATASETS.TEST[0]
        self.grouped_class_names = MetadataCatalog.get(dataset_name).thing_classes

        # Definir las transformaciones de Albumentations adecuadas para imágenes submarinas
        if is_train:
            self.augmentations = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=5, p=0.5),  # Rotaciones pequeñas para simular inclinaciones leves
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),  # Desenfoque gaussiano para simular turbidez
                # A.RandomScale(scale_limit=0.1, p=0.5),  # Eliminada para evitar recortes inválidos
                A.PadIfNeeded(min_height=800, min_width=1333, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
                A.Resize(height=800, width=1333, p=1.0),  # Redimensionar al tamaño esperado por Detectron2
                A.RandomCrop(height=800, width=1333, p=0.5),  # Recorte aleatorio
                A.Normalize(mean=(103.530, 116.280, 123.675), std=(57.375, 57.120, 58.395)),
                ToTensorV2(p=1.0)  # Convertir a tensor
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            self.augmentations = A.Compose([
                A.PadIfNeeded(min_height=800, min_width=1333, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
                A.Resize(height=800, width=1333, p=1.0),
                A.Normalize(mean=(103.530, 116.280, 123.675), std=(57.375, 57.120, 58.395)),
                ToTensorV2(p=1.0)
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

        # Inicializar contadores para la distribución de clases
        self.class_counts_before = defaultdict(int)
        self.class_counts_after = defaultdict(int)

    def __call__(self, dataset_dict):
        """
        Procesa una imagen y sus anotaciones.
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # Hacer una copia para evitar modificar el original
        # Leer la imagen
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # Mapear el category_id de cada anotación al nuevo ID de clase agrupada si se usan clases agrupadas
        annotations = []
        for ann in dataset_dict.get("annotations", []):
            original_class_name = self._get_class_name_from_id(ann["category_id"])
            if self.use_grouped_classes and original_class_name in self.class_grouping:
                new_class_name = self.class_grouping[original_class_name]
                new_class_id = self._get_new_class_id(new_class_name)
                ann["category_id"] = new_class_id
                annotations.append(ann)
                self.class_counts_before[new_class_name] += 1
            else:
                # Si no se usan clases agrupadas, mantener la clase original
                if not self.use_grouped_classes:
                    new_class_name = original_class_name
                    new_class_id = self._get_new_class_id(new_class_name)
                    ann["category_id"] = new_class_id
                    annotations.append(ann)
                    self.class_counts_before[new_class_name] += 1
                else:
                    # Si se usan clases agrupadas pero la clase no está en el agrupamiento, ignorar la anotación
                    pass

        dataset_dict["annotations"] = annotations

        # Extraer bounding boxes y etiquetas
        boxes = []
        labels = []
        for ann in dataset_dict["annotations"]:
            bbox = ann["bbox"]  # Formato COCO: [x_min, y_min, width, height]
            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann["category_id"])

        # Aplicar transformaciones de Albumentations
        transformed = self.augmentations(image=image, bboxes=boxes, labels=labels)

        image = transformed["image"]
        boxes = transformed["bboxes"]
        labels = transformed["labels"]

        # Convertir etiquetas a enteros para evitar TypeError
        labels = [int(label) for label in labels]

        # Actualizar dataset_dict con la imagen aumentada
        dataset_dict["image"] = image  # Ya está en (C, H, W)

        # Actualizar anotaciones con bounding boxes transformados
        annos = []
        for bbox, label in zip(boxes, labels):
            x_min, y_min, x_max, y_max = bbox
            annos.append({
                "bbox": [x_min, y_min, x_max, y_max],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": label
            })
            class_name = self.grouped_class_names[label]
            self.class_counts_after[class_name] += 1

        dataset_dict["annotations"] = annos

        # Convertir anotaciones a Instances
        instances = utils.annotations_to_instances(annos, image.shape[1:], mask_format="bitmask")
        dataset_dict["instances"] = instances

        return dataset_dict

    def _get_class_name_from_id(self, class_id):
        """
        Obtener el nombre de la clase a partir del ID de clase original.
        """
        return self.original_class_names[class_id]

    def _get_new_class_id(self, class_name):
        """
        Obtener el nuevo ID de clase agrupada a partir del nombre de la clase.
        """
        return self.grouped_class_names.index(class_name)

    def plot_class_distribution(self):
        """
        Graficar y guardar gráficos de barras que indican el volumen de muestras por clase antes y después del aumento de datos.
        """
        classes = list(set(list(self.class_counts_before.keys()) + list(self.class_counts_after.keys())))
        classes.sort()
        counts_before = [self.class_counts_before.get(cls, 0) for cls in classes]
        counts_after = [self.class_counts_after.get(cls, 0) for cls in classes]

        x = np.arange(len(classes))  # Posiciones de las etiquetas
        width = 0.35  # Ancho de las barras

        fig, ax = plt.subplots(figsize=(12, 8))
        rects1 = ax.bar(x - width/2, counts_before, width, label='Antes de Data Augmentation')
        rects2 = ax.bar(x + width/2, counts_after, width, label='Después de Data Augmentation')

        # Añadir texto para etiquetas, título y etiquetas personalizadas del eje x
        ax.set_ylabel('Número de Muestras')
        ax.set_title('Distribución de Clases Antes y Después de Data Augmentation')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()

        # Adjuntar una etiqueta de texto sobre cada barra en rects, mostrando su altura
        def autolabel(rects):
            """Adjuntar una etiqueta de texto sobre cada barra en *rects*, mostrando su altura."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # Desplazamiento vertical de 3 puntos
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()

        # Guardar la figura
        plot_path = os.path.join(self.output_dir, 'class_distribution_before_after_augmentation.png')
        plt.savefig(plot_path)
        plt.close()
        logging.getLogger(__name__).info(f"Class distribution plot saved to {plot_path}")
