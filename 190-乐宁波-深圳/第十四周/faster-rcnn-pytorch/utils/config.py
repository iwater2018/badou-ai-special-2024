class Config:

    def __init__(self):
        self.anchor_box_scales = [128, 256, 512]  # 3个大小
        self.anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]  # 3个宽高比
        self.rpn_stride = 16
        self.num_rois = 32  # 感兴趣区域数量
        self.verbose = True  # 是否在训练和测试过程中输出日志
        self.model_path = "logs/model.h5"
        self.rpn_min_overlap = 0.3  # 非极大值抑制的最小交并比
        self.rpn_max_overlap = 0.7  # 超过这个的认为是正样本
        self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.5
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]
