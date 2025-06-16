# prepare_dataset.py
import os
import shutil
import random
from sklearn.model_selection import train_test_split
import archive


def prepare_dataset(
    real_dir, fake_dir, output_dir="face_data", test_size=0.2, random_seed=42
):
    """
    准备伪造人脸识别数据集

    参数:
        real_dir: 真实人脸图像目录
        fake_dir: 伪造人脸图像目录
        output_dir: 输出目录
        test_size: 测试集比例
        random_seed: 随机种子
    """
    # 创建输出目录结构
    os.makedirs(os.path.join(output_dir, "train", "real"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train", "fake"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test", "real"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test", "fake"), exist_ok=True)

    # 收集真实人脸图像
    real_images = []
    for root, _, files in os.walk(real_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                real_images.append(os.path.join(root, file))

    # 收集伪造人脸图像
    fake_images = []
    for root, _, files in os.walk(fake_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                fake_images.append(os.path.join(root, file))

    print(f"找到 {len(real_images)} 张真实人脸, {len(fake_images)} 张伪造人脸")

    # 分割训练集和测试集
    real_train, real_test = train_test_split(
        real_images, test_size=test_size, random_state=random_seed
    )
    fake_train, fake_test = train_test_split(
        fake_images, test_size=test_size, random_state=random_seed
    )

    # 复制文件到对应目录
    def copy_files(files, target_dir, prefix):
        for i, src in enumerate(files):
            ext = os.path.splitext(src)[1]
            dst = os.path.join(target_dir, f"{prefix}_{i}{ext}")
            shutil.copy(src, dst)

    copy_files(real_train, os.path.join(output_dir, "train", "real"), "real")
    copy_files(real_test, os.path.join(output_dir, "test", "real"), "real")
    copy_files(fake_train, os.path.join(output_dir, "train", "fake"), "fake")
    copy_files(fake_test, os.path.join(output_dir, "test", "fake"), "fake")

    print(f"数据集准备完成！保存到 {output_dir}")
    print(f"训练集: {len(real_train)} 真实 + {len(fake_train)} 伪造")
    print(f"测试集: {len(real_test)} 真实 + {len(fake_test)} 伪造")


if __name__ == "__main__":
    # 设置数据集路径
    REAL_DIR = "archive/Human Faces Dataset/Real Images"  # 真实人脸图像目录
    FAKE_DIR = "archive/Human Faces Dataset/AI-Generated Images"  # 伪造人脸图像目录

    prepare_dataset(REAL_DIR, FAKE_DIR)
