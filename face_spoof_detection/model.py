# model.py
import torch
import torch.nn as nn


class FaceSpoofDetector(nn.Module):
    def __init__(self, input_channels=3):
        """
        伪造人脸检测模型

        参数:
            input_channels: 输入图像的通道数 (默认为3)
        """
        super(FaceSpoofDetector, self).__init__()

        # 特征提取器
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # 伪造检测特定层
        self.spoof_detector = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)
        )

    def forward(self, x):
        # 提取基本特征
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平

        # 伪造检测分支
        spoof_features = self.spoof_detector(x)

        # 分类
        output = self.classifier(spoof_features)
        return output
