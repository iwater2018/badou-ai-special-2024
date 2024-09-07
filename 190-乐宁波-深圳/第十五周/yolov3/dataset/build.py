import os

from dataset.voc import VOCDetection


def build_voc_dataset(root, img_size, data_cfg, is_train=False):
    # ------------------------- Basic parameters -------------------------
    data_dir = os.path.join(root, data_cfg['data_name'])
    num_classes = data_cfg['num_classes']
    class_names = data_cfg['class_names']
    class_indexs = data_cfg['class_indexs']
    dataset_info = {
        'num_classes': num_classes,
        'class_names': class_names,
        'class_indexs': class_indexs
    }

    # ------------------------- Build dataset -------------------------
    ## VOC dataset
    dataset = VOCDetection(
        img_size=img_size,
        data_dir=data_dir,
        image_sets=[('2007', 'trainval'), ('2012', 'trainval')] if is_train else [('2007', 'test')],
    )
    return dataset, dataset_info
