#!/usr/bin/env python
"""
Aplicación interactiva con Streamlit para cargar múltiples datasets en diferentes formatos de etiquetado,
analizar cada dataset por separado, modificar, agrupar y eliminar etiquetas, combinar los datasets y generar un nuevo archivo de anotaciones en el formato deseado.
Incluye funcionalidad para convertir anotaciones de segmentación a bounding boxes.
"""

import json
import os
import xml.etree.ElementTree as ET
import streamlit as st
from io import StringIO, BytesIO
import tempfile
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import base64

def load_coco_annotations(file):
    data = json.load(file)
    return data

def save_coco_annotations(data):
    return json.dumps(data, indent=4)

def load_pascal_voc_annotations(file):
    tree = ET.parse(file)
    root = tree.getroot()
    return root

def save_pascal_voc_annotations(root):
    tree = ET.ElementTree(root)
    with StringIO() as f:
        tree.write(f, encoding='unicode')
        return f.getvalue()

def modify_labels_coco(data, class_mapping, classes_to_delete):
    # Actualizar categorías
    original_categories = data['categories']
    new_categories = []
    category_id_mapping = {}
    new_category_id = 1

    # Filtrar categorías a eliminar
    for cat in original_categories:
        original_name = cat['name']
        if original_name in classes_to_delete:
            continue  # Omitir categorías a eliminar
        new_name = class_mapping.get(original_name, original_name)

        if new_name not in [c['name'] for c in new_categories]:
            new_categories.append({
                'id': new_category_id,
                'name': new_name,
                'supercategory': cat.get('supercategory', '')
            })
            category_id_mapping[cat['id']] = new_category_id
            new_category_id += 1
        else:
            existing_id = next(c['id'] for c in new_categories if c['name'] == new_name)
            category_id_mapping[cat['id']] = existing_id

    # Actualizar anotaciones y eliminar las de las clases a eliminar
    new_annotations = []
    for ann in data['annotations']:
        original_category_id = ann['category_id']
        original_category_name = next(cat['name'] for cat in original_categories if cat['id'] == original_category_id)
        if original_category_name in classes_to_delete:
            continue  # Omitir anotaciones de clases a eliminar
        ann['category_id'] = category_id_mapping[original_category_id]
        new_annotations.append(ann)

    data['categories'] = new_categories
    data['annotations'] = new_annotations

    # Eliminar imágenes sin anotaciones
    annotated_image_ids = set(ann['image_id'] for ann in new_annotations)
    new_images = [img for img in data['images'] if img['id'] in annotated_image_ids]
    data['images'] = new_images

    return data

def modify_labels_pascal_voc(root, class_mapping, classes_to_delete):
    objects = root.findall('object')
    for obj in objects:
        original_name = obj.find('name').text
        if original_name in classes_to_delete:
            root.remove(obj)  # Eliminar objeto de clase a eliminar
            continue
        new_name = class_mapping.get(original_name, original_name)
        obj.find('name').text = new_name

    # Si no quedan objetos en la imagen, retornar None para indicar que debe eliminarse
    if not root.findall('object'):
        return None
    return root

def merge_coco_datasets(datasets):
    merged_data = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    category_name_to_id = {}
    image_id_offset = 0
    annotation_id = 1
    category_id = 1

    for data in datasets:
        # Actualizar categorías
        for cat in data['categories']:
            cat_name = cat['name']
            if cat_name not in category_name_to_id:
                category_name_to_id[cat_name] = category_id
                merged_data['categories'].append({
                    'id': category_id,
                    'name': cat_name,
                    'supercategory': cat.get('supercategory', '')
                })
                category_id += 1

        # Actualizar imágenes y anotaciones
        image_id_mapping = {}
        for img in data['images']:
            new_image_id = img['id'] + image_id_offset
            image_id_mapping[img['id']] = new_image_id
            img['id'] = new_image_id
            merged_data['images'].append(img)

        for ann in data['annotations']:
            ann['id'] = annotation_id
            ann['image_id'] = image_id_mapping[ann['image_id']]
            ann['category_id'] = category_name_to_id[data['categories'][ann['category_id'] - 1]['name']]
            merged_data['annotations'].append(ann)
            annotation_id += 1

        image_id_offset = max(img['id'] for img in merged_data['images']) + 1

    return merged_data

def merge_pascal_voc_datasets(datasets):
    # Combinamos las raíces de los archivos XML no eliminados
    merged_roots = []
    for root in datasets:
        if root is not None:
            merged_roots.append(root)
    return merged_roots

def analyze_coco_dataset(data):
    # Número de imágenes
    num_images = len(data['images'])

    # Número de anotaciones
    num_annotations = len(data['annotations'])

    # Conteo de instancias por clase
    category_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    annotations_df = pd.DataFrame(data['annotations'])
    if not annotations_df.empty:
        annotations_df['category_name'] = annotations_df['category_id'].map(category_id_to_name)
        instances_per_class = annotations_df['category_name'].value_counts().to_dict()

        # Número de imágenes por clase
        image_ids_per_class = annotations_df.groupby('category_name')['image_id'].nunique().to_dict()
    else:
        instances_per_class = {}
        image_ids_per_class = {}

    return {
        'num_images': num_images,
        'num_annotations': num_annotations,
        'instances_per_class': instances_per_class,
        'images_per_class': image_ids_per_class
    }

def analyze_pascal_voc_dataset(root):
    if root is None:
        return {
            'num_images': 0,
            'num_annotations': 0,
            'instances_per_class': {},
            'images_per_class': {}
        }

    # Número de imágenes (1 por archivo XML)
    num_images = 1

    # Extraer objetos
    objects = root.findall('object')
    num_annotations = len(objects)

    # Conteo de instancias por clase
    class_names = [obj.find('name').text for obj in objects]
    instances_per_class = pd.Series(class_names).value_counts().to_dict()

    # Número de imágenes por clase (siempre 1 en Pascal VOC individual)
    images_per_class = {class_name: 1 for class_name in instances_per_class.keys()}

    return {
        'num_images': num_images,
        'num_annotations': num_annotations,
        'instances_per_class': instances_per_class,
        'images_per_class': images_per_class
    }

def visualize_dataset_insights(insights, dataset_name):
    st.subheader(f"Análisis del {dataset_name}")

    # Crear un contenedor para alinear las tablas
    with st.container():
        # Crear dos columnas con la misma anchura
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Número de imágenes:** {insights['num_images']}")
            st.write(f"**Número de anotaciones:** {insights['num_annotations']}")
            st.write("**Instancias por clase:**")
            st.dataframe(pd.DataFrame.from_dict(insights['instances_per_class'], orient='index', columns=['Cantidad']))

        with col2:
            st.write("")
            st.write("")
            st.write("**Número de imágenes por clase:**")
            st.dataframe(pd.DataFrame.from_dict(insights['images_per_class'], orient='index', columns=['Cantidad']))

    if insights['instances_per_class']:
        # Gráficos
        st.write("**Distribución de Instancias por Clase:**")
        fig1, ax1 = plt.subplots()
        keys = list(insights['instances_per_class'].keys())
        values = list(insights['instances_per_class'].values())
        bars = ax1.bar(range(len(keys)), values)
        ax1.set_xticks(range(len(keys)))
        ax1.set_xticklabels(keys, rotation=45, ha='right')
        ax1.set_ylabel('Cantidad de Instancias')
        ax1.set_xlabel('Clases')

        # Añadir etiquetas de datos
        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, int(yval), va='bottom', ha='center')

        st.pyplot(fig1)

    if insights['images_per_class']:
        st.write("**Distribución de Imágenes por Clase:**")
        fig2, ax2 = plt.subplots()
        keys = list(insights['images_per_class'].keys())
        values = list(insights['images_per_class'].values())
        bars = ax2.bar(range(len(keys)), values, color='orange')
        ax2.set_xticks(range(len(keys)))
        ax2.set_xticklabels(keys, rotation=45, ha='right')
        ax2.set_ylabel('Cantidad de Imágenes')
        ax2.set_xlabel('Clases')

        # Añadir etiquetas de datos
        for bar in bars:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, int(yval), va='bottom', ha='center')

        st.pyplot(fig2)

def convert_pascal_voc_to_coco(roots):
    # Función para convertir una lista de archivos Pascal VOC a formato COCO
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    category_name_to_id = {}
    category_id = 1
    annotation_id = 1
    image_id = 1

    for root in roots:
        filename = root.find('filename').text
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)

        coco_data['images'].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in category_name_to_id:
                category_name_to_id[class_name] = category_id
                coco_data['categories'].append({
                    "id": category_id,
                    "name": class_name,
                    "supercategory": ""
                })
                category_id += 1

            category_id_current = category_name_to_id[class_name]
            bndbox = obj.find('bndbox')
            bbox = [
                int(bndbox.find('xmin').text),
                int(bndbox.find('ymin').text),
                int(bndbox.find('xmax').text) - int(bndbox.find('xmin').text),
                int(bndbox.find('ymax').text) - int(bndbox.find('ymin').text)
            ]

            coco_data['annotations'].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id_current,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            })
            annotation_id += 1

        image_id += 1

    return coco_data

def convert_coco_to_pascal_voc(data):
    # Función para convertir un dataset COCO a formato Pascal VOC
    pascal_voc_files = []

    images = {img['id']: img for img in data['images']}
    categories = {cat['id']: cat['name'] for cat in data['categories']}

    for ann in data['annotations']:
        image = images[ann['image_id']]
        filename = image['file_name']
        width = image['width']
        height = image['height']
        depth = 3  # Asumimos imágenes RGB

        annotation = ET.Element('annotation')

        ET.SubElement(annotation, 'folder').text = 'images'
        ET.SubElement(annotation, 'filename').text = filename
        ET.SubElement(annotation, 'path').text = f'images/{filename}'

        source = ET.SubElement(annotation, 'source')
        ET.SubElement(source, 'database').text = 'Unknown'

        size = ET.SubElement(annotation, 'size')
        ET.SubElement(size, 'width').text = str(width)
        ET.SubElement(size, 'height').text = str(height)
        ET.SubElement(size, 'depth').text = str(depth)

        ET.SubElement(annotation, 'segmented').text = '0'

        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = categories[ann['category_id']]
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'

        bndbox = ET.SubElement(obj, 'bndbox')
        bbox = ann['bbox']
        ET.SubElement(bndbox, 'xmin').text = str(int(bbox[0]))
        ET.SubElement(bndbox, 'ymin').text = str(int(bbox[1]))
        ET.SubElement(bndbox, 'xmax').text = str(int(bbox[0] + bbox[2]))
        ET.SubElement(bndbox, 'ymax').text = str(int(bbox[1] + bbox[3]))

        pascal_voc_files.append((annotation, filename))

    return pascal_voc_files

def convert_segmentation_to_bboxes_coco(data):
    for ann in data['annotations']:
        segmentation = ann.get('segmentation', [])
        if segmentation:
            # Convertir segmentación a bounding box
            bbox = polygon_to_bbox(segmentation)
            ann['bbox'] = bbox
    return data

def polygon_to_bbox(segmentation):
    # Combinar todos los puntos del polígono
    all_points = []
    for seg in segmentation:
        xs = seg[0::2]
        ys = seg[1::2]
        all_points.extend(list(zip(xs, ys)))

    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    width = x_max - x_min
    height = y_max - y_min

    return [x_min, y_min, width, height]

def main():
    st.title("Herramienta Interactiva para Modificar y Analizar Datasets de Anotaciones")

    # Paso 1: Seleccionar formato de entrada
    st.header("Paso 1: Seleccionar Formato de Entrada")
    dataset_format = st.selectbox("Seleccione el formato de sus datasets:", ["COCO", "Pascal VOC", "COCO Segmentation"])

    # Paso 2: Cargar archivos de anotaciones
    st.header("Paso 2: Cargar Archivos de Anotaciones")
    uploaded_files = st.file_uploader(
        f"Cargue uno o más archivos de anotaciones en formato {dataset_format}",
        type=['json', 'xml'],
        accept_multiple_files=True
    )

    if uploaded_files:
        datasets = []
        label_sets = []
        dataset_insights = []

        if dataset_format == 'COCO' or dataset_format == 'COCO Segmentation':
            for idx, file in enumerate(uploaded_files):
                data = load_coco_annotations(file)

                # Si es COCO Segmentation, convertir segmentaciones a bounding boxes
                if dataset_format == 'COCO Segmentation':
                    data = convert_segmentation_to_bboxes_coco(data)

                datasets.append(data)
                label_sets.extend([cat['name'] for cat in data['categories']])

                # Analizar el dataset
                insights = analyze_coco_dataset(data)
                dataset_insights.append(insights)

                # Mostrar insights
                visualize_dataset_insights(insights, f"Dataset {idx + 1}")

        elif dataset_format == 'Pascal VOC':
            for idx, file in enumerate(uploaded_files):
                data = load_pascal_voc_annotations(file)
                datasets.append(data)
                label_sets.extend([obj.find('name').text for obj in data.findall('object')])

                # Analizar el dataset
                insights = analyze_pascal_voc_dataset(data)
                dataset_insights.append(insights)

                # Mostrar insights
                visualize_dataset_insights(insights, f"Dataset {idx + 1}")

        label_set = set(label_sets)

        # Paso 3: Modificar, agrupar o eliminar etiquetas
        st.header("Paso 3: Modificar, Agrupar o Eliminar Etiquetas")
        modify_labels = st.checkbox("¿Desea modificar o agrupar etiquetas?")
        delete_labels = st.checkbox("¿Desea eliminar etiquetas?")

        class_mapping = {}
        classes_to_delete = set()

        if modify_labels or delete_labels:
            st.info("Ingrese el nuevo nombre para cada etiqueta o seleccione las etiquetas a eliminar.")

            if delete_labels:
                classes_to_delete = st.multiselect("Seleccione las etiquetas que desea eliminar:", sorted(label_set))

            if modify_labels:
                for label in sorted(label_set):
                    if label in classes_to_delete:
                        continue  # Omitir etiquetas que se eliminarán
                    new_label = st.text_input(f"Nuevo nombre para la etiqueta '{label}':", value=label)
                    class_mapping[label] = new_label

            # Aplicar modificaciones y eliminaciones
            if dataset_format == 'COCO' or dataset_format == 'COCO Segmentation':
                modified_datasets = []
                for data in datasets:
                    data = modify_labels_coco(data, class_mapping, classes_to_delete)
                    modified_datasets.append(data)
                datasets = modified_datasets
            elif dataset_format == 'Pascal VOC':
                modified_datasets = []
                for data in datasets:
                    data = modify_labels_pascal_voc(data, class_mapping, classes_to_delete)
                    modified_datasets.append(data)
                datasets = modified_datasets

            # Re-analizar y mostrar insights actualizados
            st.header("Insights Actualizados Después de Modificaciones")
            dataset_insights = []
            if dataset_format == 'COCO' or dataset_format == 'COCO Segmentation':
                for idx, data in enumerate(datasets):
                    insights = analyze_coco_dataset(data)
                    dataset_insights.append(insights)
                    visualize_dataset_insights(insights, f"Dataset {idx + 1} (Actualizado)")
            elif dataset_format == 'Pascal VOC':
                for idx, data in enumerate(datasets):
                    insights = analyze_pascal_voc_dataset(data)
                    dataset_insights.append(insights)
                    visualize_dataset_insights(insights, f"Dataset {idx + 1} (Actualizado)")

        # Paso 4: Combinar datasets
        st.header("Paso 4: Combinar Datasets")
        combine_datasets = st.checkbox("¿Desea combinar los datasets cargados en uno solo?", value=True)

        if combine_datasets:
            if dataset_format == 'COCO' or dataset_format == 'COCO Segmentation':
                merged_data = merge_coco_datasets(datasets)
                # Analizar el dataset combinado
                combined_insights = analyze_coco_dataset(merged_data)
                st.header("Análisis del Dataset Combinado")
                visualize_dataset_insights(combined_insights, "Dataset Combinado")
            elif dataset_format == 'Pascal VOC':
                merged_data = merge_pascal_voc_datasets(datasets)
                # Analizar el dataset combinado
                total_images = len(merged_data)
                total_annotations = sum([len(root.findall('object')) for root in merged_data])
                combined_insights = {
                    'num_images': total_images,
                    'num_annotations': total_annotations,
                    'instances_per_class': {},
                    'images_per_class': {}
                }
                # Recolectar estadísticas detalladas
                class_counts = {}
                image_counts = {}
                for root in merged_data:
                    insights = analyze_pascal_voc_dataset(root)
                    for class_name, count in insights['instances_per_class'].items():
                        class_counts[class_name] = class_counts.get(class_name, 0) + count
                        image_counts[class_name] = image_counts.get(class_name, 0) + 1
                combined_insights['instances_per_class'] = class_counts
                combined_insights['images_per_class'] = image_counts

                st.header("Análisis del Dataset Combinado")
                visualize_dataset_insights(combined_insights, "Dataset Combinado")
        else:
            merged_data = datasets  # Si no se combinan, mantenemos la lista de datasets

        # Paso 5: Seleccionar formato de salida
        st.header("Paso 5: Seleccionar Formato de Salida")
        output_format = st.selectbox("Seleccione el formato de salida de las anotaciones:", ["COCO", "Pascal VOC"])

        # Paso 6: Descargar archivo de salida
        st.header("Paso 6: Descargar Archivo de Anotaciones")
        if st.button("Generar y Descargar Archivo(s) de Anotaciones"):
            if combine_datasets:
                if output_format == 'COCO':
                    if dataset_format == 'COCO' or dataset_format == 'COCO Segmentation':
                        output_data = save_coco_annotations(merged_data)
                        st.download_button(
                            label="Descargar archivo JSON combinado",
                            data=output_data,
                            file_name="annotations_combined.json",
                            mime="application/json"
                        )
                    elif dataset_format == 'Pascal VOC':
                        # Convertir a COCO
                        coco_data = convert_pascal_voc_to_coco(merged_data)
                        output_data = save_coco_annotations(coco_data)
                        st.download_button(
                            label="Descargar archivo JSON combinado",
                            data=output_data,
                            file_name="annotations_combined.json",
                            mime="application/json"
                        )
                elif output_format == 'Pascal VOC':
                    if dataset_format == 'Pascal VOC':
                        # Crear un zip con los archivos XML
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
                            for i, root in enumerate(merged_data):
                                output_data = save_pascal_voc_annotations(root)
                                file_name = f"annotation_{i+1}.xml"
                                zip_file.writestr(file_name, output_data)
                        zip_buffer.seek(0)
                        st.download_button(
                            label="Descargar archivo ZIP con anotaciones combinadas",
                            data=zip_buffer,
                            file_name="annotations_combined.zip",
                            mime="application/zip"
                        )
                    elif dataset_format == 'COCO' or dataset_format == 'COCO Segmentation':
                        pascal_voc_files = convert_coco_to_pascal_voc(merged_data)
                        # Crear un zip con los archivos XML
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                            for root, filename in pascal_voc_files:
                                output_data = save_pascal_voc_annotations(root)
                                xml_filename = filename.replace('.jpg', '.xml').replace('.png', '.xml')
                                zip_file.writestr(xml_filename, output_data)
                        zip_buffer.seek(0)
                        st.download_button(
                            label="Descargar archivo ZIP con anotaciones combinadas",
                            data=zip_buffer,
                            file_name="annotations_combined.zip",
                            mime="application/zip"
                        )
            else:
                # Si no se combinan, permitimos descargar cada dataset por separado
                for idx, data in enumerate(merged_data):
                    if output_format == 'COCO':
                        if dataset_format == 'COCO' or dataset_format == 'COCO Segmentation':
                            output_data = save_coco_annotations(data)
                            st.download_button(
                                label=f"Descargar dataset {idx+1} en formato COCO",
                                data=output_data,
                                file_name=f"annotations_{idx+1}.json",
                                mime="application/json"
                            )
                        elif dataset_format == 'Pascal VOC':
                            coco_data = convert_pascal_voc_to_coco([data])
                            output_data = save_coco_annotations(coco_data)
                            st.download_button(
                                label=f"Descargar dataset {idx+1} convertido a COCO",
                                data=output_data,
                                file_name=f"annotations_{idx+1}.json",
                                mime="application/json"
                            )
                    elif output_format == 'Pascal VOC':
                        if dataset_format == 'Pascal VOC':
                            output_data = save_pascal_voc_annotations(data)
                            st.download_button(
                                label=f"Descargar dataset {idx+1} en formato Pascal VOC",
                                data=output_data,
                                file_name=f"annotation_{idx+1}.xml",
                                mime="application/xml"
                            )
                        elif dataset_format == 'COCO' or dataset_format == 'COCO Segmentation':
                            pascal_voc_files = convert_coco_to_pascal_voc(data)
                            # Crear un zip con los archivos XML
                            zip_buffer = BytesIO()
                            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                                for root, filename in pascal_voc_files:
                                    output_data = save_pascal_voc_annotations(root)
                                    xml_filename = filename.replace('.jpg', '.xml').replace('.png', '.xml')
                                    zip_file.writestr(xml_filename, output_data)
                            zip_buffer.seek(0)
                            st.download_button(
                                label=f"Descargar dataset {idx+1} convertido a Pascal VOC",
                                data=zip_buffer,
                                file_name=f"annotations_{idx+1}.zip",
                                mime="application/zip"
                            )

    else:
        st.info("Por favor, cargue al menos un archivo de anotaciones para continuar.")

if __name__ == "__main__":
    main()
