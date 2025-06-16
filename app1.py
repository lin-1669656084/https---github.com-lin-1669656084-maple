# app.py - Flask后端应用
import os
import uuid
from flask import Flask, request, render_template, send_from_directory, jsonify
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import base64
import io
import cv2  # 新增OpenCV用于人脸检测

app = Flask(__name__)

# 全局变量存储模型
mnist_model = None  # 重命名为mnist_model
face_model = None   # 新增人脸识别模型
device = None
cfg = None
model_info = {}

# 新增人脸识别模型类
class FaceAntiSpoofingModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.classifier = torch.nn.Linear(1000, 2)  # 二分类：真实人脸 vs 伪造人脸
        
    def forward(self, x):
        features = self.base_model(x)
        return self.classifier(features)

def load_model():
    """加载训练好的模型，支持多种模型类型"""
    global mnist_model, face_model, device, cfg, model_info

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_info["models"] = {}

    # ========== 1. 加载MNIST模型 ==========
    mnist_paths = [
        ("enhanced_maple_best.pth", "enhanced"),
        ("enhanced_maple_final.pth", "enhanced"),
        ("maple_mnist_state_dict.pth", "simple"),
    ]

    mnist_loaded = False
    for model_path, model_type in mnist_paths:
        if os.path.exists(model_path):
            try:
                # ... (原有MNIST模型加载代码不变) ...
                mnist_loaded = True
                model_info["models"]["mnist"] = {
                    "type": model_info.get("type", "Unknown"),
                    "total_params": model_info.get("total_params", 0),
                    "trainable_params": model_info.get("trainable_params", 0),
                    "loaded_from": model_path,
                }
                break
            except Exception as e:
                print(f"❌ 加载MNIST模型 {model_path} 失败: {e}")
    
    # ========== 2. 加载人脸识别模型 ==========
    face_paths = [
        "face_anti_spoofing.pth",
        "face_model_resnet.pth",
        "face_model_vgg.pth"
    ]
    
    face_loaded = False
    for model_path in face_paths:
        if os.path.exists(model_path):
            try:
                # 使用预训练的ResNet作为基础模型
                from torchvision import models
                base_model = models.resnet18(pretrained=True)
                
                # 冻结基础模型参数
                for param in base_model.parameters():
                    param.requires_grad = False
                
                # 创建自定义模型
                face_model = FaceAntiSpoofingModel(base_model).to(device)
                
                # 加载训练好的权重
                checkpoint = torch.load(model_path, map_location=device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    face_model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    face_model.load_state_dict(checkpoint)
                
                face_model.eval()
                print(f"✅ 成功加载人脸识别模型: {model_path}")
                
                # 存储模型信息
                total_params = sum(p.numel() for p in face_model.parameters())
                trainable_params = sum(p.numel() for p in face_model.parameters() if p.requires_grad)
                
                model_info["models"]["face"] = {
                    "type": "FaceAntiSpoofing",
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "loaded_from": model_path
                }
                face_loaded = True
                break
                
            except Exception as e:
                print(f"❌ 加载人脸模型 {model_path} 失败: {e}")
    
    # 全局模型加载状态
    model_loaded = mnist_loaded or face_loaded
    
    if not model_loaded:
        print("⚠️ 未找到任何可用模型文件")
        print("MNIST训练命令: python enhanced_train.py 或 python train.py")
        print("人脸训练命令: python train_face.py")
    
    return model_loaded

# 启动时加载模型
model_loaded = load_model()

# ... (中间部分保持不变: 文件夹创建和MNIST预处理函数) ...

# 新增人脸图像预处理函数
def preprocess_face_image(image_data):
    """预处理人脸图像"""
    # 解析base64数据
    if "," in image_data:
        image_data = image_data.split(",")[1]
    img_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    # 使用OpenCV进行人脸检测
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        raise ValueError("未检测到人脸")
    
    # 取最大的人脸区域
    (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
    
    # 扩展人脸区域
    expand = 0.2
    x = max(0, int(x - w * expand))
    y = max(0, int(y - h * expand))
    w = min(img.width - x, int(w * (1 + 2*expand)))
    h = min(img.height - y, int(h * (1 + 2*expand)))
    
    # 裁剪人脸区域
    face_img = img.crop((x, y, x+w, y+h))
    
    # 调整大小并标准化
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    return transform(face_img).unsqueeze(0), (x, y, w, h)

# ... (中间部分保持不变: 文件路由和主页) ...

@app.route("/predict", methods=["POST"])
def predict():
    """预测API - 支持文件上传和画布数据"""
    if not model_loaded:
        return jsonify({"success": False, "error": "模型未加载，请先训练模型"})
    
    # 检查请求类型
    if "canvas_data" in request.json:
        return predict_mnist(request)
    elif "face_image" in request.json:
        return predict_face(request)
    else:
        return jsonify({"success": False, "error": "无效的请求格式"})

def predict_mnist(request):
    """处理MNIST数字识别预测"""
    if "mnist" not in model_info["models"]:
        return jsonify({"success": False, "error": "MNIST模型未加载"})
    
    try:
        # ... (原有的MNIST预测代码不变) ...
        
    except Exception as e:
        return jsonify({"success": False, "error": f"MNIST预测失败: {str(e)}"})

def predict_face(request):
    """处理人脸伪造检测预测"""
    if "face" not in model_info["models"]:
        return jsonify({"success": False, "error": "人脸模型未加载"})
    
    try:
        # 获取并预处理图像
        face_data = request.json["face_image"]
        img_tensor, face_box = preprocess_face_image(face_data)
        img_tensor = img_tensor.to(device)
        
        # 模型预测
        with torch.no_grad():
            outputs = face_model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            prediction_idx = probabilities.argmax(dim=1).item()
            confidence = probabilities[0][prediction_idx].item()
            
            # 映射预测结果
            labels = ["真实人脸", "伪造人脸"]
            prediction = labels[prediction_idx]
            
            # 获取两类概率
            real_prob = round(probabilities[0][0].item() * 100, 2)
            fake_prob = round(probabilities[0][1].item() * 100, 2)
        
        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "真实人脸": real_prob,
                "伪造人脸": fake_prob
            },
            "face_box": face_box  # 返回检测到的人脸位置
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": f"人脸检测失败: {str(e)}"})

@app.route("/model_info")
def get_model_info():
    """获取模型信息"""
    if not model_loaded:
        return jsonify({"loaded": False, "error": "模型未加载"})
    
    # 返回所有加载模型的信息
    return jsonify({
        "loaded": True,
        "models": model_info.get("models", {}),
        "device": str(device) if device else "unknown"
    })

# ... (后续代码保持不变: 训练路由、健康检查等) ...