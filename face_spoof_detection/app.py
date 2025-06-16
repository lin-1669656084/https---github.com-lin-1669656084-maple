# app.py
from flask import Flask, request, jsonify, send_from_directory
import torch
from torchvision import transforms
from PIL import Image
import os
import uuid
from model import FaceSpoofDetector
import numpy as np

app = Flask(__name__)

# 全局模型
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 加载模型
def load_model():
    global model
    if not os.path.exists("face_spoof_best.pth"):
        print("❌ 未找到训练好的模型，请先运行训练脚本")
        return False

    model = FaceSpoofDetector().to(device)
    try:
        model.load_state_dict(torch.load("face_spoof_best.pth", map_location=device))
        model.eval()
        print("✅ 伪造人脸检测模型加载完成！")
        return True
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return False


# 图像预处理
def preprocess_image(image):
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0).to(device)


# 预测端点
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"success": False, "error": "模型未加载，请先训练模型"})

    try:
        # 检查文件上传
        if "file" not in request.files:
            return jsonify({"error": "未提供文件"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "未选择文件"}), 400

        # 保存临时文件
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join("uploads", filename)
        file.save(filepath)

        # 预处理图像
        img = Image.open(filepath).convert("RGB")
        input_tensor = preprocess_image(img)

        # 预测
        with torch.no_grad():
            output = model(input_tensor)
            prob_fake = torch.sigmoid(output).item()
            is_fake = prob_fake > 0.5

        # 返回结果
        return jsonify(
            {
                "success": True,
                "is_fake": bool(is_fake),
                "confidence": round(prob_fake * 100, 2),
                "image_url": f"/uploads/{filename}",
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# 图像访问端点
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory("uploads", filename)


# 模型状态端点
@app.route("/status")
def status():
    return jsonify({"model_loaded": model is not None, "device": str(device)})


# 首页
@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>伪造人脸检测系统</title>
        <style>
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #1a2a6c, #b21f1f, #1a2a6c);
                color: #333;
                line-height: 1.6;
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 900px;
                margin: 40px auto;
                background: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
                overflow: hidden;
            }
            
            header {
                background: linear-gradient(90deg, #1a2a6c, #b21f1f);
                color: white;
                padding: 30px 20px;
                text-align: center;
            }
            
            h1 {
                font-size: 2.5rem;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            }
            
            .subtitle {
                font-size: 1.2rem;
                opacity: 0.9;
            }
            
            .content {
                padding: 30px;
            }
            
            .status-box {
                padding: 15px;
                margin: 20px 0;
                border-radius: 10px;
                text-align: center;
                font-weight: bold;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            
            .status-success {
                background: linear-gradient(90deg, #56ab2f, #a8e063);
                color: white;
            }
            
            .status-error {
                background: linear-gradient(90deg, #ff416c, #ff4b2b);
                color: white;
            }
            
            .upload-box {
                background: #f8f9fa;
                border: 2px dashed #dee2e6;
                border-radius: 10px;
                padding: 30px;
                text-align: center;
                margin: 25px 0;
                transition: all 0.3s ease;
            }
            
            .upload-box:hover {
                border-color: #6c757d;
                background: #e9ecef;
            }
            
            .file-input {
                margin: 15px 0;
                padding: 10px 20px;
                background: #1a2a6c;
                color: white;
                border: none;
                border-radius: 50px;
                cursor: pointer;
                font-size: 1rem;
                transition: all 0.3s ease;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .file-input:hover {
                background: #0d1b4d;
                transform: translateY(-2px);
                box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
            }
            
            .result-box {
                background: white;
                border-radius: 10px;
                padding: 25px;
                margin-top: 30px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
                display: none;
            }
            
            .result-content {
                text-align: center;
                padding: 20px;
            }
            
            .result-real {
                color: #28a745;
                font-size: 2rem;
                font-weight: bold;
                margin: 15px 0;
            }
            
            .result-fake {
                color: #dc3545;
                font-size: 2rem;
                font-weight: bold;
                margin: 15px 0;
            }
            
            .confidence {
                font-size: 1.2rem;
                margin: 15px 0;
                color: #6c757d;
            }
            
            .result-image {
                max-width: 300px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
                margin: 20px auto;
                display: block;
            }
            
            .btn-detect {
                background: linear-gradient(90deg, #1a2a6c, #b21f1f);
                color: white;
                border: none;
                padding: 12px 30px;
                font-size: 1.1rem;
                border-radius: 50px;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
                margin-top: 15px;
            }
            
            .btn-detect:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
            }
            
            .btn-detect:active {
                transform: translateY(0);
            }
            
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
            }
            
            .spinner {
                border: 5px solid #f3f3f3;
                border-top: 5px solid #1a2a6c;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            footer {
                text-align: center;
                padding: 20px;
                color: white;
                font-size: 0.9rem;
                margin-top: 30px;
            }
            
            @media (max-width: 768px) {
                .container {
                    margin: 20px;
                }
                
                h1 {
                    font-size: 2rem;
                }
                
                .result-image {
                    max-width: 100%;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>伪造人脸检测系统</h1>
                <p class="subtitle">基于深度学习的AI生成人脸识别解决方案</p>
            </header>
            
            <div class="content">
                <div class="status-box" id="status">
                    <!-- 状态信息将由JavaScript填充 -->
                </div>
                
                <div class="upload-box">
                    <h2>上传人脸图像进行检测</h2>
                    <p>支持 JPG, PNG 格式的图片</p>
                    <form id="uploadForm">
                        <label for="fileInput" class="file-input">
                            选择图片
                        </label>
                        <input type="file" id="fileInput" name="file" accept="image/*" hidden required>
                        <div id="fileName" style="margin: 10px 0; font-style: italic;"></div>
                        <button type="submit" class="btn-detect">检测伪造人脸</button>
                    </form>
                </div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>正在分析图像，请稍候...</p>
                </div>
                
                <div class="result-box" id="resultBox">
                    <div class="result-content" id="resultContent">
                        <!-- 结果将由JavaScript填充 -->
                    </div>
                </div>
            </div>
        </div>
        
        <footer>
            <p>© 2023 伪造人脸检测系统 | 基于AI生成人脸数据集训练</p>
        </footer>
        
        <script>
            // 更新文件名显示
            document.getElementById('fileInput').addEventListener('change', function(e) {
                const fileName = document.getElementById('fileName');
                if (this.files.length > 0) {
                    fileName.textContent = `已选择: ${this.files[0].name}`;
                } else {
                    fileName.textContent = '';
                }
            });
            
            // 检查模型状态
            function checkModelStatus() {
                fetch('/status')
                    .then(response => response.json())
                    .then(data => {
                        const statusDiv = document.getElementById('status');
                        if (data.model_loaded) {
                            statusDiv.className = 'status-box status-success';
                            statusDiv.innerHTML = '✅ 模型已加载 | 设备: ' + data.device;
                        } else {
                            statusDiv.className = 'status-box status-error';
                            statusDiv.innerHTML = '❌ 模型未加载，请先训练模型';
                        }
                    })
                    .catch(error => {
                        console.error('检查模型状态失败:', error);
                    });
            }
            
            // 页面加载时检查模型状态
            window.addEventListener('load', checkModelStatus);
            
            // 处理表单提交
            document.getElementById('uploadForm').addEventListener('submit', function(e) {
                e.preventDefault();
                
                const fileInput = document.getElementById('fileInput');
                if (!fileInput.files || fileInput.files.length === 0) {
                    alert('请选择要上传的图片');
                    return;
                }
                
                const formData = new FormData(this);
                const resultBox = document.getElementById('resultBox');
                const resultContent = document.getElementById('resultContent');
                const loading = document.getElementById('loading');
                
                // 显示加载动画，隐藏结果
                loading.style.display = 'block';
                resultBox.style.display = 'none';
                
                fetch('/predict', {
                    method: 'POST',
                    body: formData
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('网络响应异常');
                    }
                    return response.json();
                })
                .then(data => {
                    loading.style.display = 'none';
                    
                    if (data.success) {
                        resultContent.innerHTML = `
                            <h2>检测结果</h2>
                            <div class="${data.is_fake ? 'result-fake' : 'result-real'}">
                                ${data.is_fake ? 'AI生成人脸' : '真实人脸'}
                            </div>
                            <div class="confidence">
                                伪造置信度: ${data.confidence}%
                            </div>
                            <img src="${data.image_url}" alt="检测图片" class="result-image">
                        `;
                        resultBox.style.display = 'block';
                    } else {
                        alert(`检测失败: ${data.error || '未知错误'}`);
                    }
                })
                .catch(error => {
                    loading.style.display = 'none';
                    alert(`请求失败: ${error.message}`);
                });
            });
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    # 创建上传目录
    os.makedirs("uploads", exist_ok=True)

    # 加载模型
    model_loaded = load_model()

    # 启动应用
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
