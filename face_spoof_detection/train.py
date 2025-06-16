# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
from model import FaceSpoofDetector

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 数据集类
class FaceSpoofDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        """
        伪造人脸检测数据集

        参数:
            data_dir: 数据集目录 (包含real和fake子目录)
            transform: 图像变换
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        # 加载真实人脸
        real_dir = os.path.join(data_dir, "real")
        for img_name in os.listdir(real_dir):
            self.samples.append((os.path.join(real_dir, img_name), 0))

        # 加载伪造人脸
        fake_dir = os.path.join(data_dir, "fake")
        for img_name in os.listdir(fake_dir):
            self.samples.append((os.path.join(fake_dir, img_name), 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"加载图像 {img_path} 失败: {e}")
            # 返回一个替代项
            return self[(idx + 1) % len(self)]


# 训练函数
def train_model(data_dir="face_data", batch_size=32, epochs=30, lr=0.0005):
    """
    训练伪造人脸检测模型

    参数:
        data_dir: 数据集目录
        batch_size: 批次大小
        epochs: 训练轮数
        lr: 学习率
    """
    # 数据预处理
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 创建数据集
    train_dataset = FaceSpoofDataset(
        os.path.join(data_dir, "train"), transform=transform
    )
    test_dataset = FaceSpoofDataset(os.path.join(data_dir, "test"), transform=transform)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # 初始化模型
    model = FaceSpoofDetector().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer, "max", patience=3, factor=0.5, verbose=True
    )

    # 训练记录
    train_losses = []
    train_accs = []
    test_accs = []
    best_acc = 0.0

    print("开始训练伪造人脸检测模型...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 训练循环
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新进度条
            loop.set_postfix(loss=loss.item(), acc=correct / total)

        # 计算训练准确率
        train_acc = correct / total
        avg_loss = running_loss / len(train_loader)

        # 评估测试集
        test_acc = evaluate_model(model, test_loader)

        # 更新学习率
        scheduler.step(test_acc)

        # 记录指标
        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
        )

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "face_spoof_best.pth")
            print(f"✅ 保存最佳模型，准确率: {best_acc:.4f}")

    # 绘制训练曲线
    plt.figure(figsize=(15, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, "b-", linewidth=2)
    plt.title("训练损失曲线", fontsize=14, fontweight="bold")
    plt.xlabel("迭代轮次", fontsize=12)
    plt.ylabel("损失", fontsize=12)
    plt.grid(True, alpha=0.3)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, "g-", label="训练准确率", linewidth=2)
    plt.plot(test_accs, "r-", label="测试准确率", linewidth=2)
    plt.title("准确率曲线", fontsize=14, fontweight="bold")
    plt.xlabel("迭代轮次", fontsize=12)
    plt.ylabel("准确率", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=300, bbox_inches="tight")
    plt.show()

    training_time = time.time() - start_time
    print(f"训练完成！用时: {training_time//60:.0f}分{training_time%60:.0f}秒")
    print(f"最佳测试准确率: {best_acc:.4f}")


# 评估函数
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


if __name__ == "__main__":
    train_model()
