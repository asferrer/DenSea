#!/usr/bin/env python
"""
Script para agrupar clases en archivos de anotaciones COCO y generar nuevos archivos con las clases agrupadas.
"""

import json
import os
import argparse

def group_classes_in_annotations(input_json, output_json, class_grouping):
    """
    Lee el archivo de anotaciones COCO, aplica el agrupamiento de clases y guarda el nuevo archivo.

    :param input_json: Ruta al archivo de anotaciones original en formato COCO.
    :param output_json: Ruta donde se guardará el nuevo archivo de anotaciones.
    :param class_grouping: Diccionario que mapea nombres de clases originales a nombres de clases agrupadas.
    """
    with open(input_json, 'r') as f:
        data = json.load(f)

    # Crear un mapeo de id de categoría original a nuevo id de categoría
    original_categories = data['categories']
    original_id_to_name = {cat['id']: cat['name'] for cat in original_categories}
    name_to_new_name = class_grouping

    # Crear nuevas categorías agrupadas
    new_name_to_id = {}
    new_categories = []
    new_category_id = 1  # COCO suele empezar en 1

    for original_cat in original_categories:
        original_name = original_cat['name']
        if original_name in name_to_new_name:
            new_name = name_to_new_name[original_name]
            if new_name not in new_name_to_id:
                new_name_to_id[new_name] = new_category_id
                new_categories.append({
                    'id': new_category_id,
                    'name': new_name,
                    'supercategory': original_cat.get('supercategory', '')
                })
                new_category_id += 1
        else:
            # Opcional: manejar clases que no están en el agrupamiento
            pass

    # Crear mapeo de id de categoría original a nuevo id de categoría
    original_id_to_new_id = {}
    for original_id, original_name in original_id_to_name.items():
        if original_name in name_to_new_name:
            new_name = name_to_new_name[original_name]
            new_id = new_name_to_id[new_name]
            original_id_to_new_id[original_id] = new_id
        else:
            # Opcional: manejar clases que no están en el agrupamiento
            # Por ejemplo, puedes asignar un id especial o ignorarlas
            original_id_to_new_id[original_id] = None

    # Actualizar las anotaciones
    new_annotations = []
    for ann in data['annotations']:
        original_category_id = ann['category_id']
        new_category_id = original_id_to_new_id.get(original_category_id, None)
        if new_category_id is not None:
            ann['category_id'] = new_category_id
            new_annotations.append(ann)
        else:
            # Si la categoría no está en el agrupamiento, podemos decidir ignorar la anotación
            pass

    # Actualizar el campo 'categories' y 'annotations' en los datos
    data['categories'] = new_categories
    data['annotations'] = new_annotations

    # Guardar el nuevo archivo de anotaciones
    with open(output_json, 'w') as f:
        json.dump(data, f)
    print(f"Nuevo archivo de anotaciones guardado en: {output_json}")

def parse_args():
    parser = argparse.ArgumentParser(description="Agrupar clases en anotaciones COCO")
    parser.add_argument(
        "--input-json",
        required=True,
        help="Ruta al archivo de anotaciones original en formato COCO"
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Ruta donde se guardará el nuevo archivo de anotaciones"
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Definir el agrupamiento de clases
    # Mapea nombres de clases originales a nombres de clases agrupadas
    class_grouping = {
    "Can": "Small_Lightweight_Debris",
    "Squared_Can": "Small_Lightweight_Debris",
    "Bottle": "Small_Lightweight_Debris",
    "Plastic_Bag": "Small_Lightweight_Debris",
    "Packaging_Bag": "Small_Lightweight_Debris",
    "Basket": "Small_Lightweight_Debris",
    "Towel": "Textile_Debris",
    "Glove": "Textile_Debris",
    "Shoe": "Textile_Debris",
    "Plastic_Debris": "Flexible_Plastic_Debris",
    "Fishing_Net": "Flexible_Plastic_Debris",
    "Rope": "Flexible_Plastic_Debris",
    "WashingMachine": "Large_Heavy_Debris",
    "Car_Bumper": "Large_Heavy_Debris",
    "Tire": "Large_Heavy_Debris",
    "Pipe": "Large_Heavy_Debris",
    "Metal_Debris": "Small_Metal_Debris",
    "Metal_Chain": "Small_Metal_Debris",
    "Wood": "Wooden_Debris"
}

    # Llamar a la función para agrupar clases
    group_classes_in_annotations(
        input_json=args.input_json,
        output_json=args.output_json,
        class_grouping=class_grouping
    )

if __name__ == "__main__":
    main()
