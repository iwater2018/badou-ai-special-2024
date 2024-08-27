import torch

from model.yolov1 import YOLOv1

from model.loss import build_criterion


def build_yolov1(cfg, conf_thresh, nms_thresh, device, num_classes=80, trainable=False):
    print('==============================')
    print('Model Configuration: \n', cfg)

    # -------------- 构建YOLOv1 --------------
    model = YOLOv1(
        cfg=cfg,
        device=device,
        num_classes=num_classes,
        conf_thresh=conf_thresh,
        nms_thresh=nms_thresh,
        is_train=trainable
    )

    # -------------- 初始化YOLOv1的pred层参数 --------------
    # Init bias
    init_prob = 0.01
    bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
    # obj pred
    b = model.obj_pred.bias.view(1, -1)
    b.data.fill_(bias_value.item())
    model.obj_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    # cls pred
    b = model.cls_pred.bias.view(1, -1)
    b.data.fill_(bias_value.item())
    model.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    # reg pred
    b = model.reg_pred.bias.view(-1, )
    b.data.fill_(1.0)
    model.reg_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    w = model.reg_pred.weight
    w.data.fill_(0.)
    model.reg_pred.weight = torch.nn.Parameter(w, requires_grad=True)

    # -------------- 构建用于计算标签分配和计算损失的Criterion类 --------------
    criterion = None
    if trainable:
        # build criterion for training
        criterion = build_criterion(cfg, device, num_classes)

    return model, criterion
