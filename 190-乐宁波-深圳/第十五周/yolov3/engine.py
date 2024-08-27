import json

import torch.optim as optim
import torch
from collections import defaultdict
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import numpy as np


class Trainer:
    def __init__(self, model, dataloader, criterion, device, num_epochs, lr=0.001, scheduler_step_size=7,
                 scheduler_gamma=0.1, eval_dataset=None):
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size,
                                                         gamma=scheduler_gamma)
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.device = device
        self.num_epochs = num_epochs
        self.dataloader = dataloader
        self.criterion = criterion
        self.eval_dataset = eval_dataset

    def train(self, model):
        model.to(self.device)
        print(f"Starting training on {self.device}...")

        for epoch in range(self.num_epochs):
            model.train()
            running_loss = 0.0
            total_batches = len(self.dataloader)

            for i, (images, targets) in enumerate(self.dataloader):
                images = images.to(self.device)

                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=True):
                    outputs = model(images)
                    loss_dict = self.criterion(outputs, targets)
                    losses = loss_dict['losses']

                losses *= images.shape[0]
                self.scaler.scale(losses).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                running_loss += losses.item()

                if (i + 1) % 10 == 0 or (i + 1) == total_batches:
                    print(f"Epoch [{epoch + 1}/{self.num_epochs}], Step [{i + 1}/{total_batches}], "
                          f"Loss: {losses.item():.4f}")

            if self.scheduler:
                self.scheduler.step()

            epoch_loss = running_loss / total_batches
            print(f"Epoch [{epoch + 1}/{self.num_epochs}] completed. Average Loss: {epoch_loss:.4f}")

            if (epoch + 1) % 1 == 0 and self.eval_dataset is not None:
                self.evaluate_coco(model)

        print("Training complete!")

    @torch.no_grad()
    def evaluate_coco(self, model):
        # 将模型设置为评估模式
        model.eval()
        model.is_train = False

        num_images = len(self.eval_dataset)

        # 初始化 COCO 评估工具
        cocoGt = COCO('annotations.json')

        # 存储预测结果
        predictions = []

        for index in range(num_images):  # all the data in val2017
            if index % 500 == 0:
                print('[Eval: %d / %d]' % (index, num_images))

            # load an image
            img, _ = self.eval_dataset.pull_image(index)
            img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
            x = img.unsqueeze(0).to(self.device) / 255.

            bboxes, scores, labels = model(x)

            # 将输出转换为 COCO 格式
            for i, box in enumerate(bboxes):
                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])
                label = labels[i]

                bbox = [x1, y1, x2 - x1, y2 - y1]
                score = float(scores[i])  # object score * class score
                prediction = {
                    "image_id": index,
                    "category_id": int(label),
                    "bbox": bbox,
                    "score": score
                }
                predictions.append(prediction)
        model.is_train = True

        # 将预测结果写入 JSON 文件
        with open('predictions.json', 'w') as f:
            json.dump(predictions, f)
        cocoDt = cocoGt.loadRes('predictions.json')
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

        # 加载预测结果并评估
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        # 返回评估结果
        return cocoEval.stats
