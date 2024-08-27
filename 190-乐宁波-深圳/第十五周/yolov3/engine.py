class Trainer:
    def __init__(self, dataloader, criterion, device, num_epochs):
        # 初始化训练器，比如优化器、学习率调度器等
        self.optimizer = None
        self.scheduler = None
        self.device = device
        self.num_epochs = num_epochs
        self.dataloader = dataloader
        self.criterion = criterion

    def train(self, model):
        # 实现训练逻辑
        # 例如，定义训练循环，计算损失并进行反向传播，更新权重等
        for epoch in range(self.num_epochs):
            for images, targets in self.dataloader:
                images, targets = images.to(self.device), targets
                # 清零梯度
                self.optimizer.zero_grad()

                # 前向传播
                outputs = model(images)

                # 计算损失
                loss = self.criterion(outputs, targets)

                # 反向传播和优化
                loss.backward()
                self.optimizer.step()

            # 更新学习率（如果使用调度器）
            if self.scheduler:
                self.scheduler.step()

            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}")