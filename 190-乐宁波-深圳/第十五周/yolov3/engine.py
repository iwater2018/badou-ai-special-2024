import torch.optim as optim
import torch

class Trainer:
    def __init__(self, model, dataloader, criterion, device, num_epochs, lr=0.001, scheduler_step_size=7,
                 scheduler_gamma=0.1):
        # 初始化训练器，包括优化器、学习率调度器和混合精度梯度缩放器
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.device = device
        self.num_epochs = num_epochs
        self.dataloader = dataloader
        self.criterion = criterion

    def train(self, model):
        # 实现训练逻辑
        model.to(self.device)
        print(f"Starting training on {self.device}...")

        for epoch in range(self.num_epochs):
            model.train()
            running_loss = 0.0
            total_batches = len(self.dataloader)

            for i, (images, targets) in enumerate(self.dataloader):
                images = images.to(self.device)

                # 清零梯度
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=True):
                    # 前向传播
                    outputs = model(images)

                    # 计算损失
                    loss_dict = self.criterion(outputs, targets)
                    losses = loss_dict['losses']

                # 参考YOLOv5 & v8项目，损失前面要乘以batch size
                losses *= images.shape[0]

                # 计算梯度并优化
                self.scaler.scale(losses).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                running_loss += losses.item()

                if (i + 1) % 10 == 0 or (i + 1) == total_batches:
                    print(f"Epoch [{epoch + 1}/{self.num_epochs}], Step [{i + 1}/{total_batches}], "
                          f"Loss: {losses.item():.4f}")

            # 更新学习率（如果使用调度器）
            if self.scheduler:
                self.scheduler.step()

            epoch_loss = running_loss / total_batches
            print(f"Epoch [{epoch + 1}/{self.num_epochs}] completed. Average Loss: {epoch_loss:.4f}")

        print("Training complete!")
