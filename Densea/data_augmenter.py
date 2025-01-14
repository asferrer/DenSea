#!/usr/bin/env python
import os
import json
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import cv2

import albumentations as A

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd  # Importamos pandas para manejar tablas

def load_coco_annotations(annotations_file):
    coco = json.load(annotations_file)
    return coco

def save_coco_annotations(coco, save_path):
    with open(save_path, 'w') as f:
        json.dump(coco, f)
    st.success(f"Anotaciones guardadas en {save_path}")

def analyze_class_distribution(coco):
    class_counts = defaultdict(int)
    for ann in coco['annotations']:
        class_counts[ann['category_id']] += 1
    return class_counts

def get_class_to_image_ids(coco):
    class_to_images = defaultdict(set)
    for ann in coco['annotations']:
        class_to_images[ann['category_id']].add(ann['image_id'])
    return class_to_images

def get_underrepresented_classes(class_counts, desired_counts):
    underrepresented = {}
    for class_id, desired_count in desired_counts.items():
        current_count = class_counts.get(class_id, 0)
        if current_count < desired_count:
            underrepresented[class_id] = desired_count - current_count
    return underrepresented

def is_bbox_normalized(annotations, image_width, image_height):
    # Verifica si las coordenadas están normalizadas
    for ann in annotations:
        x_min, y_min, width, height = ann['bbox']
        if x_min < 0 or y_min < 0 or width < 0 or height < 0:
            return False
        if x_min <= 1 and y_min <= 1 and width <= 1 and height <=1:
            return True
    return False

def split_image(image_path, annotations, category_id_to_name, padding=50, max_padding=100):
    """
    Divide una imagen en múltiples imágenes, cada una conteniendo un solo objeto.
    Se aplica padding dinámico para incluir contexto sin solapar con otros objetos.
    """
    image = Image.open(image_path).convert('RGB')
    image_width, image_height = image.size
    cropped_data = []
    
    # Extraer todas las bounding boxes
    all_bboxes = [ann['bbox'] for ann in annotations]
    
    for ann in annotations:
        bbox = ann['bbox']
        category_id = ann['category_id']
        x_min, y_min, width, height = bbox
        x_min = int(x_min)
        y_min = int(y_min)
        width = int(width)
        height = int(height)
        x_max = x_min + width
        y_max = y_min + height

        # Inicializar padding actual
        current_padding = padding

        # Intentar aplicar el padding sin solapar otros objetos
        while current_padding <= max_padding:
            # Definir el área de recorte con padding
            crop_x_min = max(x_min - current_padding, 0)
            crop_y_min = max(y_min - current_padding, 0)
            crop_x_max = min(x_max + current_padding, image_width)
            crop_y_max = min(y_max + current_padding, image_height)

            # Verificar si el recorte solapa con otras bounding boxes
            overlap = False
            for other_ann in annotations:
                if other_ann == ann:
                    continue
                other_bbox = other_ann['bbox']
                ox_min, oy_min, owidth, oheight = other_bbox
                ox_min = int(ox_min)
                oy_min = int(oy_min)
                owidth = int(owidth)
                oheight = int(oheight)
                ox_max = ox_min + owidth
                oy_max = oy_min + oheight

                # Calcular intersección
                inter_x_min = max(crop_x_min, ox_min)
                inter_y_min = max(crop_y_min, oy_min)
                inter_x_max = min(crop_x_max, ox_max)
                inter_y_max = min(crop_y_max, oy_max)

                if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
                    overlap = True
                    break

            if not overlap:
                # No hay solapamiento, se puede recortar con este padding
                break
            else:
                # Reducir el padding y probar nuevamente
                current_padding -= 10
                if current_padding < 0:
                    # No es posible aplicar padding sin solapar
                    break

        # Definir el área de recorte final
        final_padding = current_padding if current_padding >=0 else 0
        crop_x_min = max(x_min - final_padding, 0)
        crop_y_min = max(y_min - final_padding, 0)
        crop_x_max = min(x_max + final_padding, image_width)
        crop_y_max = min(y_max + final_padding, image_height)

        # Recortar la imagen
        cropped_image = image.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))

        # Ajustar la bounding box relativa al nuevo recorte
        new_x_min = x_min - crop_x_min
        new_y_min = y_min - crop_y_min
        new_bbox = [new_x_min, new_y_min, width, height]

        # Crear nueva anotación
        new_ann = {
            'bbox': new_bbox,
            'category_id': category_id,
            'area': new_bbox[2] * new_bbox[3],
            'iscrowd': ann.get('iscrowd', 0),
            'segmentation': []  # Añade segmentación si es necesario
        }

        cropped_data.append({
            'image': cropped_image,
            'annotation': new_ann,
            'category_id': category_id,
            'category_name': category_id_to_name.get(category_id, str(category_id))
        })

    return cropped_data

def augment_image(cropped_image, annotation, transforms, image_id, ann_id_start):
    """
    Aplica transformaciones de aumento a una imagen recortada y ajusta la anotación.
    """
    image_np = np.array(cropped_image)
    bbox_coco = annotation['bbox']
    category_id = annotation['category_id']

    # Convertir bbox de formato COCO a Pascal VOC
    x_min, y_min, width, height = bbox_coco
    x_max = x_min + width
    y_max = y_min + height
    bbox_pascal = [x_min, y_min, x_max, y_max]

    # Aplicar las transformaciones
    transformed = transforms(image=image_np, bboxes=[bbox_pascal], category_ids=[category_id])

    # Verificar si alguna bounding box es inválida
    if len(transformed['bboxes']) == 0:
        raise ValueError("Después de la transformación, no queda ninguna bounding box válida.")

    augmented_image = Image.fromarray(transformed['image'])

    new_bbox_pascal = transformed['bboxes'][0]
    new_category_id = transformed['category_ids'][0]

    # Validar las coordenadas de la bounding box
    x_min_new, y_min_new, x_max_new, y_max_new = new_bbox_pascal
    if x_max_new <= x_min_new or y_max_new <= y_min_new:
        raise ValueError(f"Bounding box inválida después de la transformación: {new_bbox_pascal}")

    # Convertir bounding box de Pascal VOC a COCO
    width_new = x_max_new - x_min_new
    height_new = y_max_new - y_min_new
    coco_bbox = [x_min_new, y_min_new, width_new, height_new]

    new_ann = {
        'id': ann_id_start,
        'image_id': image_id,
        'category_id': new_category_id,
        'bbox': coco_bbox,
        'area': width_new * height_new,
        'iscrowd': annotation.get('iscrowd', 0),
        'segmentation': []  # Añade segmentación si es necesario
    }

    ann_id_start += 1

    return augmented_image, new_ann, ann_id_start

def draw_bounding_box(image, annotation, category_id_to_name):
    """
    Dibuja una bounding box en la imagen.
    """
    draw = ImageDraw.Draw(image)
    bbox_coco = annotation['bbox']
    category_id = annotation['category_id']
    x_min, y_min, width, height = bbox_coco
    x_max = x_min + width
    y_max = y_min + height

    draw.rectangle([(x_min, y_min), (x_max, y_max)], outline='red', width=2)
    draw.text((x_min, y_min), category_id_to_name.get(category_id, str(category_id)), fill='white')
    return image

def plot_class_distribution(class_counts_before, class_counts_after, desired_counts, category_id_to_name, title_before, title_after):
    classes = list(desired_counts.keys())
    classes_sorted = sorted(classes, key=lambda x: category_id_to_name.get(x, x))
    class_names = [category_id_to_name.get(cls, str(cls)) for cls in classes_sorted]
    counts_before = [class_counts_before.get(cls, 0) for cls in classes_sorted]
    counts_after = [class_counts_after.get(cls, 0) for cls in classes_sorted]

    plt.figure(figsize=(14, 8))
    x = np.arange(len(classes_sorted))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width/2, counts_before, width, label='Antes del Aumento de Datos', color='skyblue')
    rects2 = ax.bar(x + width/2, counts_after, width, label='Después del Aumento de Datos', color='salmon')

    ax.set_ylabel('Número de Anotaciones')
    ax.set_title('Distribución de Clases Antes y Después del Aumento de Datos')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()

    def autolabel(rects):
        """Adjuntar una etiqueta de texto sobre cada barra en *rects*, mostrando su altura."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(int(height)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # Desplazamiento vertical de 3 puntos
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

def augment_dataset(coco, images_dir, save_dir, desired_counts, category_id_to_name, transforms):
    # Mantener las imágenes y anotaciones originales
    augmented_coco = deepcopy(coco)
    
    # Analizar distribución de clases antes del aumento
    original_class_counts = analyze_class_distribution(augmented_coco)
    class_counts = deepcopy(original_class_counts)
    underrepresented_classes = get_underrepresented_classes(class_counts, desired_counts)
    st.write(f"**Clases subrepresentadas (necesitan imágenes adicionales):**")
    
    # Crear diccionarios para acceso rápido
    annotations_dict = defaultdict(list)
    for ann in augmented_coco['annotations']:
        annotations_dict[ann['image_id']].append(ann)
    
    # Crear un mapa de clase a imágenes que la contienen
    class_to_images = get_class_to_image_ids(augmented_coco)
    
    # Manejar clases sin imágenes base
    classes_to_remove = []
    for class_id in underrepresented_classes.keys():
        images_with_class = list(class_to_images.get(class_id, []))
        if not images_with_class:
            class_name = category_id_to_name.get(class_id, str(class_id))
            st.warning(f"No hay imágenes disponibles para la clase {class_name}. Se omitirá esta clase.")
            classes_to_remove.append(class_id)
    # Eliminar clases sin imágenes base
    for class_id in classes_to_remove:
        del underrepresented_classes[class_id]
    
    # Actualizar lista de clases subrepresentadas
    if not underrepresented_classes:
        st.info("¡Todas las clases ya cumplen con el número deseado de imágenes o no hay imágenes base disponibles para aumentar!")
        return augmented_coco, []
    
    # Actualizar total de aumentaciones necesarias
    total_needed = sum(underrepresented_classes.values())
    progress_bar = st.progress(0)
    current_progress = 0
    
    # Contadores para IDs únicos
    max_image_id = max([img['id'] for img in augmented_coco['images']], default=0)
    max_ann_id = max([ann['id'] for ann in augmented_coco['annotations']], default=0)
    new_image_id = max_image_id + 1
    new_ann_id = max_ann_id + 1
    
    # Lista para guardar ejemplos de imágenes aumentadas
    example_images = []
    
    # Crear directorio para imágenes aumentadas
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Crear directorio para guardar ejemplos
    example_dir = save_dir_path / "examples"
    example_dir.mkdir(exist_ok=True)
    
    # Inicializar diccionario para contar aumentaciones por clase
    augmentation_counts = {class_id: 0 for class_id in underrepresented_classes.keys()}
    
    # Máximo de aumentaciones por objeto
    max_augmentations_per_object = 4
    
    # Crear contenedores en Streamlit para la tabla de progreso
    st.write("### Progreso de Aumentación por Clase")
    progress_table_placeholder = st.empty()
    
    # Función para actualizar la tabla de progreso
    def update_progress_table():
        data = []
        for class_id in underrepresented_classes.keys():
            class_name = category_id_to_name.get(class_id, str(class_id))
            original_count = original_class_counts.get(class_id, 0)
            augmented_count = augmentation_counts[class_id]
            total_count = original_count + augmented_count
            desired_count = desired_counts[class_id]
            data.append({
                'Clase': class_name,
                'Original': original_count,
                'Aumentadas': augmented_count,
                'Total Actual': total_count,
                'Deseado': desired_count,
                'Progreso': f"{(total_count / desired_count) * 100:.2f}%"
            })
        df = pd.DataFrame(data)
        progress_table_placeholder.table(df)
    
    # Mostrar tabla inicial
    update_progress_table()
    
    # Procesar cada clase subrepresentada
    for class_id in underrepresented_classes.keys():
        class_name = category_id_to_name.get(class_id, str(class_id))
        needed = underrepresented_classes[class_id]
        current_count = original_class_counts[class_id]
        desired_count = desired_counts[class_id]
        images_with_class = list(class_to_images[class_id])
        
        # Obtener todas las anotaciones de la clase
        all_annotations_of_class = [ann for ann in augmented_coco['annotations'] if ann['category_id'] == class_id]
        num_objects = len(all_annotations_of_class)
        
        if num_objects == 0:
            st.warning(f"No hay anotaciones disponibles para la clase {class_name}. Se omitirá esta clase.")
            continue
        
        # Inicializar contador de aumentaciones por objeto
        object_augmentation_counts = defaultdict(int)
        
        # Mezclar las anotaciones para distribuir las aumentaciones
        random.shuffle(all_annotations_of_class)
        
        idx = 0
        # Mientras no se alcance el número deseado de imágenes
        while current_count < desired_count:
            ann = all_annotations_of_class[idx % num_objects]
            idx += 1
            
            # Verificar si se alcanzó el máximo de aumentaciones para este objeto
            if object_augmentation_counts[ann['id']] >= max_augmentations_per_object:
                continue
            
            # Obtener la imagen correspondiente
            image_id = ann['image_id']
            img = next((img for img in augmented_coco['images'] if img['id'] == image_id), None)
            if img is None:
                st.warning(f"Imagen con ID {image_id} no encontrada.")
                continue
            image_path = Path(images_dir) / img['file_name']
            if not image_path.is_file():
                st.warning(f"Imagen no encontrada: {image_path}")
                continue
            image_annotations = annotations_dict[image_id]
            
            # Verificar si las bounding boxes están normalizadas
            normalized = is_bbox_normalized(image_annotations, img['width'], img['height'])
            if normalized:
                # Convertir a coordenadas absolutas
                for ann_i in image_annotations:
                    ann_i['bbox'] = [
                        ann_i['bbox'][0] * img['width'],
                        ann_i['bbox'][1] * img['height'],
                        ann_i['bbox'][2] * img['width'],
                        ann_i['bbox'][3] * img['height']
                    ]
            
            # Recortar el objeto si ya se alcanzó el máximo de aumentaciones y hay más de un objeto en la imagen
            if object_augmentation_counts[ann['id']] == max_augmentations_per_object:
                if len(image_annotations) > 1:
                    # Crear un nuevo recorte del objeto sin otros objetos en el fondo
                    try:
                        cropped_data = split_image(image_path, image_annotations, category_id_to_name)
                        # Encontrar el recorte correspondiente a la clase actual
                        for data in cropped_data:
                            if data['category_id'] == class_id:
                                cropped_image = data['image']
                                new_ann = data['annotation']
                                break
                        else:
                            st.warning(f"No se encontró el objeto de la clase {class_name} en la imagen recortada.")
                            continue
                    except Exception as e:
                        st.warning(f"Error al recortar la imagen {image_path}: {e}")
                        continue
                else:
                    # No se puede recortar más, continuar al siguiente objeto
                    continue
            else:
                # Usar la imagen completa
                try:
                    cropped_image = Image.open(image_path).convert('RGB')
                    new_ann = ann.copy()
                except Exception as e:
                    st.warning(f"Error al abrir la imagen {image_path}: {e}")
                    continue
            
            # Aplicar augmentación
            try:
                augmented_image, augmented_ann, new_ann_id = augment_image(
                    cropped_image, new_ann, transforms, new_image_id, new_ann_id)
            except Exception as e:
                st.warning(f"Error al aumentar la imagen {image_path}: {e}")
                continue
            
            # Guardar la imagen aumentada
            new_image_name = f"{image_path.stem}_aug_{new_image_id}{image_path.suffix}"
            augmented_image.save(save_dir_path / new_image_name)
            
            # Crear la nueva entrada de imagen
            new_image_info = {
                'id': new_image_id,
                'width': augmented_image.width,
                'height': augmented_image.height,
                'file_name': new_image_name
            }
            augmented_coco['images'].append(new_image_info)
            augmented_coco['annotations'].append(augmented_ann)
            
            # Actualizar contadores
            new_image_id += 1
            current_count += 1
            class_counts[class_id] += 1
            augmentation_counts[class_id] += 1
            object_augmentation_counts[ann['id']] += 1
            current_progress += 1
            progress_bar.progress(min(current_progress / total_needed, 1.0))
            
            # Actualizar tabla de progreso
            update_progress_table()
            
            # Guardar ejemplos para visualización (hasta 5)
            if len(example_images) < 5:
                img_with_box = draw_bounding_box(augmented_image.copy(), augmented_ann, category_id_to_name)
                example_images.append(img_with_box)
                img_with_box.save(example_dir / new_image_name)
            
    # Gráfica después del aumento
    class_counts_after = analyze_class_distribution(augmented_coco)
    plot_class_distribution(original_class_counts, class_counts_after, desired_counts, category_id_to_name, 
                            "Distribución de Clases Antes del Aumento de Datos",
                            "Distribución de Clases Después del Aumento de Datos")
    
    return augmented_coco, example_images

def main():
    st.title("Generador de Dataset Aumentado para Detección de Residuos Submarinos")
    st.write("""
        Este aplicativo permite aplicar aumentos de datos a un dataset en formato COCO, enfocándose en equilibrar todas las clases
        asegurando que cada una tenga un número deseado de imágenes mediante aumentos dinámicos. 
        Además, al recortar las imágenes, se asegura de incluir contexto de fondo sin solapar con otros objetos.
    """)

    st.sidebar.header("Configuración de Aumento de Datos")

    annotations_file = st.sidebar.file_uploader("Sube el archivo de anotaciones COCO (annotations.json)", type=['json'])
    images_dir = st.sidebar.text_input("Ruta al directorio de imágenes originales", value="/ruta/a/tu/dataset/images")
    save_dir = st.sidebar.text_input("Ruta al directorio donde se guardarán las imágenes aumentadas y anotaciones", value="/ruta/a/tu/nuevo_dataset_aumentado")
    desired_images_per_class = st.sidebar.number_input("Número deseado de imágenes por clase", min_value=1, value=100)
    
    if st.sidebar.button("Iniciar Aumento de Datos"):
        if not annotations_file or not images_dir or not save_dir:
            st.error("Por favor, completa todos los campos requeridos.")
        else:
            images_dir_path = Path(images_dir)
            save_dir_path = Path(save_dir)
            # La ruta para guardar las anotaciones ahora se construye automáticamente
            annotations_filename = annotations_file.name
            save_annotations_path = save_dir_path / annotations_filename

            if not images_dir_path.is_dir():
                st.error(f"El directorio de imágenes no existe: {images_dir}")
            else:
                # Crear directorio donde se guardarán las imágenes aumentadas
                save_dir_path.mkdir(parents=True, exist_ok=True)

                # Crear directorio para guardar ejemplos
                example_dir = save_dir_path / "examples"
                example_dir.mkdir(exist_ok=True)

                st.write("Iniciando el proceso de aumento de datos...")

                # Cargar anotaciones COCO
                try:
                    coco = load_coco_annotations(annotations_file)
                except Exception as e:
                    st.error(f"Error al cargar el archivo de anotaciones: {e}")
                    return

                # Obtener nombres de clases
                categories = coco.get('categories', [])
                if not categories:
                    st.error("El archivo de anotaciones no contiene categorías.")
                    return
                category_id_to_name = {cat['id']: cat['name'] for cat in categories}

                # Crear un diccionario con el número deseado de imágenes por clase
                desired_counts = {cat['id']: desired_images_per_class for cat in categories}

                # Definir transformaciones de aumento de datos adecuadas para imágenes submarinas
                transforms = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, p=0.5),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                    A.MotionBlur(blur_limit=7, p=0.3),
                    A.MedianBlur(blur_limit=7, p=0.3),
                    A.ToGray(p=0.1),
                    A.Resize(height=800, width=1333, p=1.0),
                    A.RandomCrop(height=800, width=1333, p=0.5),
                    A.PadIfNeeded(min_height=800, min_width=1333, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
                ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

                # Aumentar el dataset
                try:
                    augmented_coco, example_images = augment_dataset(
                        coco,
                        images_dir_path,
                        save_dir_path,
                        desired_counts,
                        category_id_to_name,
                        transforms
                    )
                except Exception as e:
                    st.error(f"Error durante el aumento de datos: {e}")
                    return

                # Guardar nuevas anotaciones
                try:
                    save_coco_annotations(augmented_coco, save_annotations_path)
                except Exception as e:
                    st.error(f"Error al guardar las nuevas anotaciones: {e}")
                    return

                # Mostrar ejemplos de imágenes aumentadas
                if example_images:
                    st.header("Ejemplos de Imágenes Aumentadas")
                    cols = st.columns(2)
                    for idx, img in enumerate(example_images):
                        with cols[idx % 2]:
                            st.image(img, caption=f"Imagen Aumentada {idx+1}", use_container_width=True)
                else:
                    st.info("No se generaron ejemplos de imágenes aumentadas.")

                st.success("Proceso completado exitosamente.")
                st.balloons()

if __name__ == "__main__":
    main()
