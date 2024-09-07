import os.path as osp
import random
import torch.utils.data as data
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import torch

from dataset.data_augment.ssd_augment import Resize

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


class VOCAnnotationTransform(object):
    def __init__(self, class_to_ind=None):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt if i % 2 == 0 else cur_pt
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [x1, y1, x2, y2, label_ind]

        return res  # [[x1, y1, x2, y2, label_ind], ... ]


class VOCDetection(data.Dataset):
    def __init__(self,
                 img_size=640,
                 data_dir=None,
                 image_sets=(('2007', 'trainval'), ('2012', 'trainval')),
                 is_train=False
                 ):
        self.root = data_dir
        self.img_size = img_size
        self.image_set = image_sets
        self.target_transform = VOCAnnotationTransform()
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        self.is_train = is_train
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        image, target = self.pull_item(index)
        return image, target

    def __len__(self):
        return len(self.ids)

    def load_image_target(self, index):
        img_id = self.ids[index]
        image = cv2.imread(self._imgpath % img_id)
        height, width, _ = image.shape

        anno = ET.parse(self._annopath % img_id).getroot()
        if self.target_transform:
            anno = self.target_transform(anno)

        target = {
            "boxes": np.array(anno).reshape(-1, 5)[:, :4],
            "labels": np.array(anno).reshape(-1, 5)[:, 4],
            "orig_size": [height, width]
        }

        image, target['boxes'], target['labels'] = Resize(self.img_size)(
            image, target['boxes'].copy(), target['labels'].copy()
        )

        return torch.from_numpy(image).permute(2, 0, 1).contiguous().float(), {
            "boxes": torch.from_numpy(target['boxes']).float(),
            "labels": torch.from_numpy(target['labels']).float(),
            "orig_size": target['orig_size'],
            "img_id": img_id,
        }

    def pull_item(self, index):
        image, target = self.load_image_target(index)
        return image, target

    def pull_image(self, index):
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR), img_id

    def pull_anno(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno)
        return img_id[1], gt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='VOC-Dataset')

    # opt
    parser.add_argument('--root', default='/Users/liuhaoran/Desktop/python_work/object-detection/dataset/VOCdevkit/',
                        help='data root')

    args = parser.parse_args()

    trans_config = {
        'aug_type': 'yolov5',  # 或者改为'ssd'来使用SSD风格的数据增强
        # Basic Augment
        'degrees': 0.0,  # 可以修改数值来决定旋转图片的程度，如改为YOLOX默认的10.0
        'translate': 0.2,  # 可以修改数值来决定平移图片的程度，
        'scale': [0.1, 2.0],  # 图片尺寸扰动的比例范围
        'shear': 0.0,  # 可以修改数值来决定旋转图片的程度，如改为YOLOX默认的2.0
        'perspective': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        # Mosaic & Mixup
        'mosaic_prob': 1.0,  # 使用马赛克增强的概率：0～1
        'mixup_prob': 1.0,  # 使用混合增强的概率：0～1
        'mosaic_type': 'yolov5_mosaic',
        'mixup_type': 'yolox_mixup',  # 或者改为'yolov5_mixup'，使用yolov5风格的混合增强
        'mixup_scale': [0.5, 1.5]
    }

    dataset = VOCDetection(
        img_size=args.img_size,
        data_dir=args.root,
        is_train=args.is_train
    )

    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(20)]
    print('Data length: ', len(dataset))

    for i in range(1000):
        image, target = dataset.pull_item(i)
        # to numpy
        image = image.permute(1, 2, 0).numpy()
        # to uint8
        image = image.astype(np.uint8)
        image = image.copy()
        img_h, img_w = image.shape[:2]

        boxes = target["boxes"]
        labels = target["labels"]

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            if x2 - x1 > 1 and y2 - y1 > 1:
                cls_id = int(label)
                color = class_colors[cls_id]
                # class name
                label = VOC_CLASSES[cls_id]
                image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                # put the test on the bbox
                cv2.putText(image, label, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
        cv2.imshow('gt', image)
        # cv2.imwrite(str(i)+'.jpg', img)
        cv2.waitKey(0)
