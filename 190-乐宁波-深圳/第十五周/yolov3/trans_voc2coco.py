import json
import os
import xml.etree.ElementTree as ET


def convert_voc_to_coco(annotations_dir, images_dir, output_file):
    anns = []
    images = []
    categories = [{'id': 1, 'name': 'class_name'}]  # 根据实际情况添加类别

    for filename in os.listdir(images_dir):
        if not filename.endswith('.jpg'):
            continue

        image_id = len(images)
        image_path = os.path.join(images_dir, filename)
        image = {
            'file_name': filename,
            'height': 0,
            'width': 0,
            'id': image_id
        }
        images.append(image)

        anno_path = os.path.join(annotations_dir, filename.replace('jpg', 'xml'))
        tree = ET.parse(anno_path)
        root = tree.getroot()

        image['height'] = int(root.find('size').find('height').text)
        image['width'] = int(root.find('size').find('width').text)

        for obj in root.iter('object'):
            ann = {
                'area': 0,  # 根据需要计算
                'bbox': [0, 0, 0, 0],  # 根据需要计算
                'category_id': 1,  # 根据实际情况设置
                'id': len(anns) + 1,
                'image_id': image_id,
                'iscrowd': 0
            }
            anns.append(ann)

    coco_format = {
        'images': images,
        'annotations': anns,
        'categories': categories
    }

    with open(output_file, 'w') as f:
        json.dump(coco_format, f)


root = 'D:\dataset'
annotations_dir = os.path.join(root, 'VOCdevkit/VOC2007/Annotations')
images_dir = os.path.join(root, 'VOCdevkit/VOC2007/JPEGImages')
output_file = 'annotations.json'
convert_voc_to_coco(annotations_dir, images_dir, output_file)
