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

app = Flask(__name__)

# 全局变量存储模型
model = None
device = None
cfg = None
model_info = {}


def load_model():
    """加载训练好的模型，支持多种模型类型"""
    global model, device, cfg, model_info

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型文件优先级列表
    model_paths = [
        ("enhanced_maple_best.pth", "enhanced"),
        ("enhanced_maple_final.pth", "enhanced"),
        ("maple_mnist_state_dict.pth", "simple")
    ]

    for model_path, model_type in model_paths:
        if os.path.exists(model_path):
            try:
                if model_type == "enhanced":
                    # 尝试导入增强模型
                    try:
                        from enhanced_maple import EnhancedMaPLe, EnhancedConfig
                        cfg = EnhancedConfig()
                        model = EnhancedMaPLe(
                            input_dim=28 * 28,
                            num_classes=10,
                            prompt_dim=cfg.prompt_dim,
                            mask_ratio=cfg.mask_ratio,
                            num_layers=cfg.num_layers,
                            dropout_rate=cfg.dropout_rate,
                            use_attention=cfg.use_attention,
                            use_residual=cfg.use_residual
                        ).to(device)

                        # 加载模型权重
                        checkpoint = torch.load(model_path, map_location=device)
                        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['model_state_dict'])
                            if 'best_accuracy' in checkpoint:
                                model_info['best_accuracy'] = checkpoint['best_accuracy']
                        else:
                            model.load_state_dict(checkpoint)

                        model_info['type'] = 'EnhancedMaPLe'
                        print(f"✅ 成功加载增强模型: {model_path}")

                    except ImportError:
                        print("❌ 无法导入增强模型，尝试简单模型...")
                        continue

                elif model_type == "simple":
                    # 使用你提供的原始MaPLe模型
                    import torch.nn as nn

                    class MaPLe(nn.Module):
                        def __init__(self, input_dim=28 * 28, num_classes=10, prompt_dim=100, mask_ratio=0.5):
                            super(MaPLe, self).__init__()
                            self.prompt_dim = prompt_dim
                            self.mask_ratio = mask_ratio
                            self.embedding = nn.Linear(input_dim, prompt_dim)
                            self.prompt_mask = nn.Parameter(torch.ones(prompt_dim), requires_grad=True)
                            self.classifier = nn.Linear(prompt_dim, num_classes)

                        def forward(self, x):
                            B = x.size(0)
                            x = x.view(B, -1)  # flatten
                            x = self.embedding(x)  # project to prompt space

                            # Apply learnable mask
                            mask = torch.sigmoid(self.prompt_mask)  # values between 0 and 1
                            masked_x = x * mask  # element-wise multiplication

                            logits = self.classifier(masked_x)
                            return logits

                    model = MaPLe(prompt_dim=100, mask_ratio=0.5).to(device)
                    model.load_state_dict(torch.load(model_path, map_location=device))

                    # 创建简单配置
                    class SimpleConfig:
                        def __init__(self):
                            self.prompt_dim = 100
                            self.mask_ratio = 0.5
                            self.num_layers = 1
                            self.dropout_rate = 0.0

                    cfg = SimpleConfig()
                    model_info['type'] = 'SimpleMaPLe'
                    print(f"✅ 成功加载简单模型: {model_path}")

                model.eval()

                # 计算模型参数信息
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                model_info.update({
                    'total_params': total_params,
                    'trainable_params': trainable_params,
                    'device': str(device),
                    'loaded_from': model_path
                })

                return True

            except Exception as e:
                print(f"❌ 加载模型 {model_path} 失败: {e}")
                continue

    print("⚠️ 未找到可用的模型文件，请先训练模型")
    print("运行命令: python enhanced_train.py 或 python train.py")
    return False


# 启动时加载模型
model_loaded = load_model()

# 创建必要的文件夹
UPLOAD_FOLDER = 'uploads'
TEMPLATE_FOLDER = 'templates'
for folder in [UPLOAD_FOLDER, TEMPLATE_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)


def preprocess_image(image_path=None, image_data=None):
    """预处理图像，支持文件路径或base64数据"""
    if image_path:
        img = Image.open(image_path).convert('L')
    elif image_data:
        # 处理base64数据
        if ',' in image_data:
            image_data = image_data.split(',')[1]  # 移除data:image/png;base64,前缀
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes)).convert('L')
    else:
        raise ValueError("必须提供image_path或image_data")

    # 调整大小到28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # 转换为numpy数组
    img_array = np.array(img)

    # 检查是否需要反转颜色（MNIST是黑底白字）
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    # 标准化到[0,1]范围
    img_array = img_array.astype(np.float32) / 255.0

    # 转换为tensor并添加batch维度
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)

    return img_tensor


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """提供上传文件的访问"""
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/')
def index():
    """主页 - 直接返回HTML内容"""
    # 如果没有templates文件夹或模板文件，直接返回HTML
    template_path = os.path.join(TEMPLATE_FOLDER, 'enhanced_index.html')

    if not os.path.exists(template_path):
        # 创建模板文件
        html_content = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MaPLe 手写数字识别</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { text-align: center; }
        .status { padding: 10px; margin: 10px; border-radius: 5px; }
        .success { background-color: #d4edda; color: #155724; }
        .error { background-color: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 MaPLe 手写数字识别</h1>
        <div class="status {status_class}">
            {status_message}
        </div>
        <p>请访问完整版界面或通过API接口使用模型</p>
        <div>
            <h3>API接口:</h3>
            <ul>
                <li>POST /predict - 预测接口</li>
                <li>GET /model_info - 模型信息</li>
            </ul>
        </div>
    </div>
</body>
</html>"""

        status_class = "success" if model_loaded else "error"
        status_message = "✅ 模型已加载" if model_loaded else "❌ 模型未加载，请先训练模型"

        return html_content.format(
            status_class=status_class,
            status_message=status_message
        )

    try:
        return render_template('enhanced_index.html', model_loaded=model_loaded)
    except:
        # 备用简单页面
        return f"""
        <html>
            <head><title>MaPLe 手写数字识别</title></head>
            <body style="font-family: Arial; text-align: center; padding: 50px;">
                <h1>🧠 MaPLe 手写数字识别</h1>
                <p>模型状态: {'✅ 已加载' if model_loaded else '❌ 未加载'}</p>
                <p>请将上面提供的HTML代码保存为 templates/enhanced_index.html</p>
            </body>
        </html>
        """


@app.route('/predict', methods=['POST'])
def predict():
    """预测API - 支持文件上传和画布数据"""
    if not model_loaded:
        return jsonify({
            'success': False,
            'error': '模型未加载，请先训练模型'
        })

    try:
        # 处理文件上传
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename != '':
                # 保存文件
                filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)

                # 预处理图像
                img_tensor = preprocess_image(image_path=filepath)
                image_url = f'/uploads/{filename}'
            else:
                return jsonify({'success': False, 'error': '没有选择文件'})

        # 处理画布数据
        elif request.is_json and 'canvas_data' in request.json:
            canvas_data = request.json['canvas_data']
            img_tensor = preprocess_image(image_data=canvas_data)
            image_url = None

        else:
            return jsonify({'success': False, 'error': '没有提供图像数据'})

        # 模型预测
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            outputs = model(img_tensor)

            # 处理不同的输出格式
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            # 计算概率和预测
            probabilities = F.softmax(logits, dim=1)
            prediction = logits.argmax(dim=1).item()
            confidence = probabilities.max().item()

            # 获取所有类别的概率
            all_probs = probabilities[0].cpu().numpy().tolist()

        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'confidence': round(confidence * 100, 2),
            'probabilities': [round(p * 100, 2) for p in all_probs],
            'image_url': image_url
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'预测失败: {str(e)}'
        })


@app.route('/model_info')
def get_model_info():
    """获取模型信息"""
    if not model_loaded:
        return jsonify({
            'loaded': False,
            'error': '模型未加载'
        })

    return jsonify({
        'loaded': True,
        'model_type': model_info.get('type', 'Unknown'),
        'total_parameters': model_info.get('total_params', 0),
        'trainable_parameters': model_info.get('trainable_params', 0),
        'device': model_info.get('device', 'unknown'),
        'loaded_from': model_info.get('loaded_from', 'unknown'),
        'best_accuracy': model_info.get('best_accuracy', 'N/A'),
        'config': {
            'prompt_dim': cfg.prompt_dim if cfg else 'N/A',
            'mask_ratio': cfg.mask_ratio if cfg else 'N/A',
            'num_layers': getattr(cfg, 'num_layers', 'N/A'),
            'dropout_rate': getattr(cfg, 'dropout_rate', 'N/A')
        }
    })


@app.route('/train', methods=['POST'])
def start_training():
    """启动训练提示"""
    return jsonify({
        'success': True,
        'message': '请在命令行运行训练脚本',
        'instructions': [
            '增强模型: python enhanced_train.py',
            '简单模型: python train.py',
            '然后重新启动Flask应用'
        ]
    })


@app.route('/health')
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model_loaded,
        'device': str(device) if device else 'unknown',
        'timestamp': str(torch.initial_seed())
    })


@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({
        'error': 'Not Found',
        'message': '请访问 / 获取主页'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    return jsonify({
        'error': 'Internal Server Error',
        'message': '服务器内部错误'
    }), 500


if __name__ == '__main__':
    print("🚀 启动Flask应用...")
    print(f"📱 模型加载状态: {'✅ 成功' if model_loaded else '❌ 失败'}")

    if model_loaded:
        print(f"🧠 模型类型: {model_info.get('type', 'Unknown')}")
        print(f"💾 参数数量: {model_info.get('total_params', 0):,}")
        print(f"🖥️  设备: {model_info.get('device', 'unknown')}")
    else:
        print("⚠️  请先运行以下命令训练模型:")
        print("   python enhanced_train.py  # 增强模型")
        print("   python train.py          # 简单模型")

    print("🌐 访问 http://127.0.0.1:5000 开始使用")
    print("📡 API接口:")
    print("   POST /predict     - 预测接口")
    print("   GET  /model_info  - 模型信息")
    print("   GET  /health      - 健康检查")

    app.run(debug=True, host='127.0.0.1', port=5000)