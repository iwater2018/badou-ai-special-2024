import os.path
from utils.config import Config


class FRCNN:
    model_path = 'model_data/voc_weights.h5'
    classes_path = 'model_data/voc_classes.txt'
    confidence = 0.7

    def __init__(self):
        self.class_names = self._get_class()
        self.config = Config()
        self.generate()
        # self.bbox_util = BBoxUtility()

    def _get_class(self):
        """获取所有分类"""
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5')

        # 总的种类数量, +1是为了考虑背景类
        self.num_classes = len(self.class_names) + 1

        self.model_rpn, self.model_classifier = frcnn.get_predict_model(self.config, self.num_classes)
        self.model_rpn.load_weights(self.model_path, by_name=True)