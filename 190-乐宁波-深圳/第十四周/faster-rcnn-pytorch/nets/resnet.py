import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size):
        super(IdentityBlock, self).__init__()
        filters1, filters2, filters3 = filters

        self.conv2a = nn.Conv2d(in_channels, filters1, kernel_size=(1, 1))
        self.bn2a = nn.BatchNorm2d(filters1)

        self.conv2b = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, padding='same')
        self.bn2b = nn.BatchNorm2d(filters2)

        self.conv2c = nn.Conv2d(filters2, filters3, kernel_size=(1, 1))
        self.bn2c = nn.BatchNorm2d(filters3)

    def forward(self, x):
        shortcut = x

        x = self.conv2a(x)
        x = self.bn2a(x)
        x = F.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x)
        x = F.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x)

        x += shortcut
        x = F.relu(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, strides=(2, 2)):
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = filters

        self.conv2a = nn.Conv2d(in_channels, filters1, kernel_size=(1, 1), stride=strides)
        self.bn2a = nn.BatchNorm2d(filters1)

        self.conv2b = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, padding='same')
        self.bn2b = nn.BatchNorm2d(filters2)

        self.conv2c = nn.Conv2d(filters2, filters3, kernel_size=(1, 1))
        self.bn2c = nn.BatchNorm2d(filters3)

        self.shortcut_conv = nn.Conv2d(in_channels, filters3, kernel_size=(1, 1), stride=strides)
        self.shortcut_bn = nn.BatchNorm2d(filters3)

    def forward(self, x):
        shortcut = self.shortcut_conv(x)
        shortcut = self.shortcut_bn(shortcut)

        x = self.conv2a(x)
        x = self.bn2a(x)
        x = F.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x)
        x = F.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x)

        x += shortcut
        x = F.relu(x)
        return x


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.bn1 = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.layer1 = nn.Sequential(
            ConvBlock(64, [64, 64, 256], kernel_size=3, strides=(1, 1)),
            IdentityBlock(256, [64, 64, 256], kernel_size=3),
            IdentityBlock(256, [64, 64, 256], kernel_size=3),
        )

        self.layer2 = nn.Sequential(
            ConvBlock(256, [128, 128, 512], kernel_size=3),
            IdentityBlock(512, [128, 128, 512], kernel_size=3),
            IdentityBlock(512, [128, 128, 512], kernel_size=3),
            IdentityBlock(512, [128, 128, 512], kernel_size=3),
        )

        self.layer3 = nn.Sequential(
            ConvBlock(512, [256, 256, 1024], kernel_size=3),
            IdentityBlock(1024, [256, 256, 1024], kernel_size=3),
            IdentityBlock(1024, [256, 256, 1024], kernel_size=3),
            IdentityBlock(1024, [256, 256, 1024], kernel_size=3),
            IdentityBlock(1024, [256, 256, 1024], kernel_size=3),
            IdentityBlock(1024, [256, 256, 1024], kernel_size=3),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


if __name__ == "__main__":
    model = ResNet50()
    input_tensor = torch.randn(1, 3, 600, 600)
    output = model(input_tensor)
    print(output.shape)
    print(model)
