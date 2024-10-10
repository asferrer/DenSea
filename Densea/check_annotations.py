import json
import os
import pycocotools.mask as mask_utils
from tqdm import tqdm
from collections import defaultdict

def convert_and_validate_bboxes(coco_annotation_file, output_annotation_file, min_width=10, min_height=10, max_aspect_ratio=10.0):
    """
    Convierte las segmentaciones en las anotaciones COCO a bounding boxes,
    verifica las etiquetas, valida las bounding boxes y genera un reporte.

    Args:
        coco_annotation_file (str): Ruta al archivo de anotaciones COCO con segmentaciones.
        output_annotation_file (str): Ruta donde se guardará el nuevo archivo de anotaciones con bounding boxes.
        min_width (int): Ancho mínimo permitido para las bounding boxes.
        min_height (int): Alto mínimo permitido para las bounding boxes.
        max_aspect_ratio (float): Máximo ratio de aspecto permitido (ancho/alto o alto/ancho).
    """
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)

    # Verificar que el archivo tiene los campos necesarios
    required_fields = ['info', 'licenses', 'images', 'annotations', 'categories']
    for field in required_fields:
        if field not in coco_data:
            print(f"El campo '{field}' no está presente en el archivo de anotaciones.")
            return

    annotations = coco_data['annotations']
    images = {img['id']: img for img in coco_data['images']}
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Diccionarios para análisis
    class_counts = defaultdict(int)
    invalid_categories = set()
    invalid_bboxes = []

    # Recorrer todas las anotaciones y actualizar las bounding boxes
    for ann in tqdm(annotations, desc="Procesando anotaciones"):
        segmentation = ann.get('segmentation', None)
        category_id = ann.get('category_id', None)
        image_id = ann.get('image_id', None)

        # Verificar que el category_id es válido
        if category_id not in categories:
            print(f"Anotación ID {ann['id']} tiene un category_id inválido: {category_id}")
            invalid_categories.add(category_id)
            continue

        # Verificar que la imagen existe
        if image_id not in images:
            print(f"Anotación ID {ann['id']} tiene un image_id inválido: {image_id}")
            continue

        # Contar la anotación para la clase correspondiente
        class_counts[category_id] += 1

        if segmentation is None:
            print(f"Anotación ID {ann['id']} no tiene segmentación.")
            continue

        if isinstance(segmentation, list):
            # Segmentación en formato de polígonos
            # Crear una máscara binaria a partir del polígono
            rles = mask_utils.frPyObjects(segmentation, images[image_id]['height'], images[image_id]['width'])
            rle = mask_utils.merge(rles)
        elif isinstance(segmentation, dict):
            # Segmentación en formato RLE
            rle = segmentation
        else:
            print(f"Formato de segmentación desconocido en anotación ID {ann['id']}.")
            continue

        # Obtener la máscara binaria
        mask = mask_utils.decode(rle)

        # Encontrar las coordenadas de la bounding box a partir de la máscara
        ys, xs = mask.nonzero()
        if len(xs) == 0 or len(ys) == 0:
            print(f"Máscara vacía en anotación ID {ann['id']}.")
            continue

        x_min = xs.min()
        x_max = xs.max()
        y_min = ys.min()
        y_max = ys.max()
        width = x_max - x_min + 1  # +1 porque los índices son inclusivos
        height = y_max - y_min + 1

        # Verificar que la bounding box cumple con los criterios
        bbox_valid = True
        reasons = []

        # Verificar tamaño mínimo
        if width < min_width or height < min_height:
            bbox_valid = False
            reasons.append("Bounding box demasiado pequeña")

        # Verificar aspecto
        aspect_ratio = max(width / height, height / width)
        if aspect_ratio > max_aspect_ratio:
            bbox_valid = False
            reasons.append("Aspect ratio de bounding box fuera de rango")

        # Verificar que la bounding box está dentro de la imagen
        img_info = images[image_id]
        img_width = img_info['width']
        img_height = img_info['height']
        if x_min < 0 or y_min < 0 or x_max >= img_width or y_max >= img_height:
            bbox_valid = False
            reasons.append("Bounding box fuera de los límites de la imagen")

        if not bbox_valid:
            invalid_bboxes.append({
                "annotation_id": ann['id'],
                "image_id": image_id,
                "reasons": reasons
            })
            continue  # O puedes optar por corregir las bounding boxes aquí

        # Actualizar la bounding box en la anotación
        ann['bbox'] = [float(x_min), float(y_min), float(width), float(height)]
        ann['area'] = float(width * height)

        # Asegurar que 'iscrowd' está definido
        if 'iscrowd' not in ann:
            ann['iscrowd'] = 0  # Valor por defecto

    # Generar el reporte de clases
    print("\nReporte de volumen de muestras por clase:")
    for cat_id, count in class_counts.items():
        class_name = categories.get(cat_id, "Categoría desconocida")
        print(f"- {class_name} (ID {cat_id}): {count} muestras")

    if invalid_categories:
        print("\nCategorías inválidas encontradas en las anotaciones:")
        for cat_id in invalid_categories:
            print(f"- category_id inválido: {cat_id}")

    if invalid_bboxes:
        print(f"\nAnotaciones con bounding boxes inválidas: {len(invalid_bboxes)}")
        for bbox_info in invalid_bboxes:
            ann_id = bbox_info['annotation_id']
            img_id = bbox_info['image_id']
            reasons = ", ".join(bbox_info['reasons'])
            print(f"- Anotación ID {ann_id}, Imagen ID {img_id}: {reasons}")

    # Guardar el nuevo archivo de anotaciones en formato COCO
    with open(output_annotation_file, 'w') as f:
        json.dump(coco_data, f)
    print(f"\nNuevo archivo de anotaciones guardado en {output_annotation_file}")
if __name__ == "__main__":
    # Ajusta las rutas según tus archivos
    coco_annotation_file = r"G:\Mi unidad\Densea\DiffusionDet\datasets\synthetic\v2\test_coco\annotations.json"
    output_annotation_file = r"G:\Mi unidad\Densea\DiffusionDet\datasets\synthetic\v2\test_coco\annotations_bbox.json"

    # Parámetros de validación (ajústalos según tus necesidades)
    min_width = 10
    min_height = 10
    max_aspect_ratio = 100.0

    convert_and_validate_bboxes(coco_annotation_file, output_annotation_file, min_width, min_height, max_aspect_ratio)
    from pycocotools.coco import COCO

    try:
        coco = COCO(output_annotation_file)
        print("El archivo de anotaciones es válido según COCO API.")
    except Exception as e:
        print(f"Error al validar el archivo de anotaciones: {e}")

