import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from data_manager import get_dataloaders
from maple import MaPLe
from utils import evaluate
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import os

def train():
    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    model = MaPLe(prompt_dim=cfg.prompt_dim, mask_ratio=cfg.mask_ratio).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    train_loader, test_loader = get_dataloaders(cfg.batch_size)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch + 1} | Loss: {total_loss:.4f} | Test Accuracy: {acc * 100:.2f}%")
    torch.save(model.state_dict(), "maple_mnist_state_dict.pth")

# 从自定义路径加载图片并预测
def predict_from_path(model, image_path, device):
    # 定义图像预处理函数
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    # 打开图片
    img = Image.open(image_path).convert('L')
    # 预处理图片
    img = transform(img).unsqueeze(0).to(device)

    # 进行预测
    with torch.no_grad():
        logits = model(img)
        pred = logits.argmax(dim=1).item()
    return pred

if __name__ == "__main__":
    train()

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 重新创建模型结构
    model = MaPLe(prompt_dim=cfg.prompt_dim, mask_ratio=cfg.mask_ratio).to(device)
    model.load_state_dict(torch.load("maple_mnist_state_dict.pth", map_location=device))
    model.eval()

    # 自定义图片路径
    custom_image_path = input("请输入要预测的图片路径: ")
    if os.path.exists(custom_image_path):
        prediction = predict_from_path(model, custom_image_path, device)
        print(f"预测结果: {prediction}")
    else:
        print("输入的路径不存在，请检查后重新输入。")

    # 推理例子：测试集前5张图像
    from data_manager import get_dataloaders

    _, test_loader = get_dataloaders(batch_size=5)
    images, labels = next(iter(test_loader))
    images = images.to(device)

    with torch.no_grad():
        logits = model(images)
        preds = logits.argmax(dim=1)

    print("Ground truth:", labels.tolist())
    print("Predicted:", preds.cpu().tolist())

    for i in range(5):
        plt.imshow(images[i].cpu().squeeze(), cmap='gray')
        plt.title(f"Label: {labels[i]} | Pred: {preds[i].item()}")
        plt.axis('off')
        plt.show()