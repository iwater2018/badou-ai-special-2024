import numpy as np
import torch
from torch import nn

from model.backbone import build_backbone
from model.neck import build_neck
from model.head import build_head


class YOLOv1(nn.Module):
    def __init__(self, cfg, device, num_classes, is_train, conf_thresh, nms_thresh):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        self.is_train = is_train
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.stride = 32

        # 主干网络
        self.backbone, feat_dim = build_backbone(cfg['backbone'], is_train & cfg['pretrained'])

        # 颈部网络
        self.neck = build_neck(cfg, feat_dim, out_dim=512)
        head_dim = self.neck.out_dim

        # 检测头
        self.head = build_head(cfg, head_dim, head_dim, num_classes)

        # 预测层
        self.obj_pred = nn.Conv2d(head_dim, 1, kernel_size=1)
        self.cls_pred = nn.Conv2d(head_dim, num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(head_dim, 4, kernel_size=1)

    def forward(self, x):
        if not self.is_train:
            return self.inference(x)

        feat = self.backbone(x)

        feat = self.neck(feat)

        cls_feat, reg_feat = self.head(feat)

        obj_pred = self.obj_pred(cls_feat)
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)
        fmp_size = obj_pred.shape[-2:]  # 这是指最终预测特征图的height和width

        # 调整pred
        # [B,C,H,W] -> [B,H,W,C] -> [B,H*W,C]
        obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

        # 计算边界框在原图坐标
        box_pred = self.decode_boxes(reg_pred, fmp_size)

        # 网络输出
        outputs = {"pred_obj": obj_pred,  # (torch.Tensor) [B, M, 1]
                   "pred_cls": cls_pred,  # (torch.Tensor) [B, M, C]
                   "pred_box": box_pred,  # (torch.Tensor) [B, M, 4]
                   "stride": self.stride,  # (Int)
                   "fmp_size": fmp_size  # (List[int, int])
                   }
        return outputs

    def inference(self, x):
        feat = self.backbone(x)

        feat = self.neck(feat)

        cls_feat, reg_feat = self.head(feat)

        obj_pred = self.obj_pred(cls_feat)
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)
        fmp_size = obj_pred.shape[-2:]  # 这是指最终预测特征图的height和width

        # 调整pred
        # [B,C,H,W] -> [B,H,W,C] -> [B,H*W,C]
        obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

        # 推理阶段一般batch默认为1
        obj_pred = obj_pred[0]
        cls_pred = cls_pred[0]
        reg_pred = reg_pred[0]

        # 计算边界框在原图坐标
        bboxes = self.decode_boxes(reg_pred, fmp_size)

        # 计算边界框得分
        scores = torch.sqrt((obj_pred.sigmoid() * cls_pred.sigmoid()))
        # 放在CPU上
        scores = scores.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # 后处理
        bboxes, scores, labels = self.postprocess(bboxes, scores)

        return bboxes, scores, labels

    def create_grid(self, input_size):
        # 特征图宽高
        ws, hs = input_size

        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])

        # to [H,W,2]
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        # [H,W,2]->[HW,2]
        grid_xy = grid_xy.view(-1, 2).to(self.device)
        return grid_xy

    def decode_boxes(self, pred_reg, fmp_size):
        grid_cell = self.create_grid(fmp_size)

        # 计算预测框在原图尺寸下的中心点
        pred_ctr = (torch.sigmoid(pred_reg[..., :2]) + grid_cell) * self.stride
        pred_wh = torch.exp(pred_reg[..., 2:]) * self.stride

        pred_x1y1 = pred_ctr - pred_wh * 0.5
        pred_x2y2 = pred_ctr + pred_wh * 0.5
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box

    def postprocess(self, bboxes, scores):
        # 选出得分最高的类别标签
        labels = np.argmax(scores, axis=1)

        scores = scores[(np.arange(scores.shape[0]), labels)]  # 解释一下这一行

        # 过滤低于阈值的
        keep = np.where(scores > self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # nms非极大值抑制
        keep = np.zeros(len(bboxes), dtype=np.int32)
        for i in range(self.num_classes):
            indexs = np.where(labels == i)[0]
            if len(indexs) == 0:
                continue
            c_bboxes = bboxes[indexs]
            c_scores = scores[indexs]
            c_keep = self.nms(c_bboxes, c_scores, self.nms_thresh)
            keep[indexs[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        return bboxes, scores, labels

    def nms(self, bboxes, scores, nms_thresh):
        if len(bboxes) == 0:
            return []

        # 计算每个框的区域
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

        # 获取排序后的索引，scores.argsort()从小到大排序的索引，[::-1]表示翻转
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # 计算交集的左上角和右下角点坐标
            xx1 = np.maximum(bboxes[i, 0], bboxes[order[1:], 0])
            yy1 = np.maximum(bboxes[i, 1], bboxes[order[1:], 1])
            xx2 = np.minimum(bboxes[i, 2], bboxes[order[1:], 2])
            yy2 = np.minimum(bboxes[i, 3], bboxes[order[1:], 3])

            # 计算交集的宽高
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            # 计算交集、并集面积
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter
            # 计算交并比
            iou = inter / union
            # 滤除超过NMS阈值的边界框
            inds = np.where(iou <= nms_thresh)[0]
            order = order[inds + 1]  # inds+1才是剩余框的索引
        return keep


class YoloMatcher(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    @torch.no_grad()
    def __call__(self, fmp_size, stride, targets):
        """
        输入参数的解释:
            img_size: (Int) 输入图像的尺寸
            stride:   (Int) YOLOv1网络的输出步长
            targets:  (List[Dict]) 为List类型，包含一批数据的标签，每一个数据标签为Dict类型，其主要的数据结构为：
                             dict{'boxes':  (torch.Tensor) [N, 4], 一张图像中的N个目标边界框坐标
                                  'labels': (torch.Tensor) [N,], 一张图像中的N个目标类别标签
                                  ...}
        """
        # 准备后续处理会用到的变量
        bs = len(targets)
        fmp_h, fmp_w = fmp_size
        gt_objectness = np.zeros([bs, fmp_h, fmp_w, 1])
        gt_classes = np.zeros([bs, fmp_h, fmp_w, self.num_classes])
        gt_bboxes = np.zeros([bs, fmp_h, fmp_w, 4])

        # 第一层for循环遍历每一张图像的标签
        for batch_index in range(bs):
            targets_per_image = targets[batch_index]
            # [N,]
            tgt_cls = targets_per_image["labels"].numpy()
            # [N, 4]
            tgt_box = targets_per_image['boxes'].numpy()

            # 第二层for循环遍历该张图像的每一个目标的标签
            for gt_box, gt_label in zip(tgt_box, tgt_cls):
                # 获得该目标的边界框坐标
                x1, y1, x2, y2 = gt_box

                # 计算目标框的中心点坐标和宽高
                xc, yc = (x2 + x1) * 0.5, (y2 + y1) * 0.5
                bw, bh = x2 - x1, y2 - y1

                # 检查该目标边界框是否有效
                if bw < 1. or bh < 1.:
                    continue

                    # 计算中心点所在的网格坐标
                xs_c = xc / stride
                ys_c = yc / stride
                grid_x = int(xs_c)
                grid_y = int(ys_c)

                #  检查网格坐标是否有效
                if grid_x < fmp_w and grid_y < fmp_h:
                    # 标记objectness标签，即此处的网格有物体，对应一个正样本
                    gt_objectness[batch_index, grid_y, grid_x] = 1.0

                    # 标记正样本处的类别标签，采用one-hot格式
                    cls_ont_hot = np.zeros(self.num_classes)
                    cls_ont_hot[int(gt_label)] = 1.0
                    gt_classes[batch_index, grid_y, grid_x] = cls_ont_hot

                    # 标记正样本处的bbox标签
                    gt_bboxes[batch_index, grid_y, grid_x] = np.array([x1, y1, x2, y2])

        # 将标签数据的shape从 [B, H, W, C] 的形式reshape成 [B, M, C] ，其中M = HW，以便后续的处理
        gt_objectness = gt_objectness.reshape(bs, -1, 1)
        gt_classes = gt_classes.reshape(bs, -1, self.num_classes)
        gt_bboxes = gt_bboxes.reshape(bs, -1, 4)

        # 将numpy.array类型转换为torch.Tensor类型
        gt_objectness = torch.from_numpy(gt_objectness).float()
        gt_classes = torch.from_numpy(gt_classes).float()
        gt_bboxes = torch.from_numpy(gt_bboxes).float()

        return gt_objectness, gt_classes, gt_bboxes
