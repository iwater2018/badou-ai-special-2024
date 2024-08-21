from __future__ import absolute_import
from __future__ import division
import torch as t
import numpy as np
from utils import array_tool as at
from model.utils.bbox_tools import loc2bbox
from torchvision.ops import nms
# from model.utils.nms import non_maximum_suppression

from torch import nn
from data.dataset import preprocess
from torch.nn import functional as F
from utils.config import opt


def nograd(f):
    def new_f(*args, **kwargs):
        with t.no_grad():
            return f(*args, **kwargs)

    return new_f


class FasterRCNN(nn.Module):
    """
    这是Faster R-CNN的基类。

    这是支持对象检测API [#]_ 的Faster R-CNN链接的基类。Faster R-CNN由以下三个阶段组成：

    1. **Feature extraction**: 获取图像并计算它们的特征图。
    2. **Region Proposal Networks**: 根据上一阶段计算出的特征图，生成围绕对象的一系列[感兴趣区域](RoIs)
    3. **Localization and Classification Heads**: 使用属于提议RoIs的特征图，对RoIs中的对象进行类别分类并改进定位。

    每个阶段由一个可调用的 torch.nn.Module 对象：feature、rpn 和 head 来执行。

    有两个函数 predict 和 __call__ 用于进行对象检测。
    :meth:`predict`函数接收图像并返回转换为图像坐标的边界框。这在将Faster R-CNN视为一个黑盒函数的情况下非常有用，例如。
    :meth:`__call__`函数提供在需要中间输出的场景下使用，例如在训练和调试时。

    支持对象检测API的链接具有相同接口的 predict 方法。更多详情请参阅 predict。

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        extractor (nn.Module): 一个模块，它接收一个BCHW图像数组并返回特征图。
        rpn (nn.Module): 一个模块，它具有与 model.region_proposal_network.RegionProposalNetwork 类相同的接口。
            请参阅那里的文档。
        head (nn.Module): 一个模块，它接收一个BCHW变量、RoIs和RoIs的批量索引。这返回类依赖的定位参数和类分数。
        loc_normalize_mean (tuple of four floats): 定位估计的均值。
        loc_normalize_std (tupler of four floats): 定位估计的标准差。
    """

    def __init__(self, extractor, rpn, head,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)
                 ):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head

        # mean and std
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('evaluate')

    @property
    def n_class(self):
        # 包括背景在内的总类别数
        return self.head.n_class

    def forward(self, x, scale=1.):
        """执行Faster R-CNN的前向传播。

        缩放参数 :obj:`scale` 被RPN用来确定选择小对象的阈值，这些小对象将被拒绝，无论它们的置信度分数如何。

        以下是使用的符号。

        * :math:`N` 是批量大小的数字
        * :math:`R'` 是跨批量产生的感兴趣区域（RoIs）的总数。给定 :math:`i` 第 :math:`i` 幅图像提出的 :math:`R_i` 个RoIs，:math:`R' = \\sum _{i=1} ^ N R_i`.
        * :math:`L` 是不包括背景的类别数。

        类别按背景、第1类、...、第 :math:`L` 类的顺序排列。

        参数：
            x (autograd.Variable): 4D图像变量。
            scale (float): 在预处理期间应用于原始图像的缩放量。

        返回：
            Variable, Variable, array, array:
            返回以下四个值的元组。

            * **roi_cls_locs**: 所提议RoIs的偏移量和缩放。其形状是 :math:`(R', (L + 1) \\times 4)`。
            * **roi_scores**: 对提议RoIs的类别预测。其形状是 :math:`(R', L + 1)`。
            * **rois**: 由RPN提出的RoIs。其形状是 :math:`(R', 4)`。
            * **roi_indices**: RoIs的批量索引。其形状是 :math:`(R',)`。

        """
        img_size = x.shape[2:]

        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.rpn(h, img_size, scale)
        roi_cls_locs, roi_scores = self.head(
            h, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices

    def use_preset(self, preset):
        """
        在预测期间使用给定的预设。

        此方法更改 :obj:`self.nms_thresh` 和 :obj:`self.score_thresh` 的值。
        这些值分别是用于非极大值抑制的阈值和在 :meth:`predict` 中丢弃低置信度提议的阈值。

        如果需要将属性更改为预设中未提供的其他值，请通过直接访问公共属性进行修改。

        参数：
            preset ({'visualize', 'evaluate'}): 一个字符串，用于确定要使用的预设。
        """
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = nms(cls_bbox_l, prob_l, self.nms_thresh)
            # import ipdb;ipdb.set_trace()
            # keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    @nograd
    def predict(self, imgs, sizes=None, visualize=False):
        """Detect objects from images.

        This method predicts objects for each image.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.

        Returns:
           tuple of lists:
           This method returns a tuple of three lists,
           :obj:`(bboxes, labels, scores)`.

           * **bboxes**: A list of float arrays of shape :math:`(R, 4)`, \
               where :math:`R` is the number of bounding boxes in a image. \
               Each bouding box is organized by \
               :math:`(y_{min}, x_{min}, y_{max}, x_{max})` \
               in the second axis.
           * **labels** : A list of integer arrays of shape :math:`(R,)`. \
               Each value indicates the class of the bounding box. \
               Values are in range :math:`[0, L - 1]`, where :math:`L` is the \
               number of the foreground classes.
           * **scores** : A list of float arrays of shape :math:`(R,)`. \
               Each value indicates how confident the prediction is.

        """
        self.eval()
        if visualize:
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(at.tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
            prepared_imgs = imgs
        bboxes = list()
        labels = list()
        scores = list()
        for img, size in zip(prepared_imgs, sizes):
            img = at.totensor(img[None]).float()
            scale = img.shape[3] / size[1]
            roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale)
            # We are assuming that batch size is 1.
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            roi = at.totensor(rois) / scale

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = t.Tensor(self.loc_normalize_mean).cuda(). \
                repeat(self.n_class)[None]
            std = t.Tensor(self.loc_normalize_std).cuda(). \
                repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
                                at.tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = at.totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            # clip bounding box
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

            prob = (F.softmax(at.totensor(roi_score), dim=1))

            bbox, label, score = self._suppress(cls_bbox, prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.use_preset('evaluate')
        self.train()
        return bboxes, labels, scores

    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify 
        special optimizer
        """
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if opt.use_adam:
            self.optimizer = t.optim.Adam(params)
        else:
            self.optimizer = t.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer
