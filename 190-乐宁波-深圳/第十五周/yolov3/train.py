from copy import deepcopy

import torch

from config.dataset_config import dataset_cfg
from config.yolov1_config import yolov1_cfg
from dataset.build import build_voc_dataset
from engine import Trainer
from model.build import build_yolov1
from utils.misc import build_dataloader, compute_flops, CollateFunc


def train(is_cuda, num_workers, batch_size, root, img_size):
    # 如果args.cuda为True，则使用GPU来训练，否则使用CPU来训练（强烈不推荐）
    if is_cuda:
        print('use GPU to train')
        device = torch.device("cuda")
    else:
        print('use CPU to train')
        device = torch.device("cpu")

    dataset, dataset_info = build_voc_dataset(root, img_size, dataset_cfg['voc'])
    num_classes = dataset_info['num_classes']

    dataloader = build_dataloader(num_workers, dataset, batch_size, collate_fn=CollateFunc())

    # 构建YOLO模型
    model, criterion = build_yolov1(yolov1_cfg, 0.005, 0.6, device, num_classes, is_train=True)

    # 将模型切换至train模式
    model = model.to(device).train()

    # 标记单卡模式的model，方便我们做一些其他的处理，省去了DDP模式下的model.module的判断
    model_without_ddp = model

    # 计算模型的参数量和FLOPs

    model_copy = deepcopy(model_without_ddp)
    model_copy.trainable = False
    model_copy.eval()
    compute_flops(model=model_copy,
                  img_size=img_size,
                  device=device)
    del model_copy

    # 构建训练所需的Trainer类
    trainer = Trainer(model, dataloader, criterion, device, num_epochs=100)

    # 开始训练我们的模型
    trainer.train(model)
    # --------------------------------- Train: End ---------------------------------

    # 训练完毕后，清空占用的GPU显存
    del trainer
    if is_cuda:
        torch.cuda.empty_cache()


if __name__ == '__main__':
    train(is_cuda=True, num_workers=2, batch_size=4, root='D:\dataset', img_size=7 * 32)
