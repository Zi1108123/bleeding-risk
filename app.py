from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# 加载训练好的随机森林模型
model = joblib.load("bleeding_risk_model.pkl")

# 初始化 Flask 应用
app = Flask(__name__)

@app.route('/')
def home():
    """显示主页"""
    return render_template('index.html')  # 创建简单网页模板

@app.route('/predict', methods=['POST'])
def predict():
    """处理预测请求"""
    try:
        # 从请求获取输入特征
        data = request.json  # 如果是通过 API 请求
        features = np.array(data['features']).reshape(1, -1)  # 转换为模型输入格式
        
        # 执行预测
        probability = model.predict_proba(features)[0, 1]  # 获取正类概率
        prediction = model.predict(features)[0]  # 获取预测类别
        
        # 返回预测结果
        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/web-predict', methods=['POST'])
def web_predict():
    """通过网页表单处理预测请求"""
    try:
        # 从表单中获取数据
        features = [float(request.form[f'feature_{i}']) for i in range(10)]  # 假设有10个特征
        features = np.array(features).reshape(1, -1)  # 转换为模型输入格式

        # 执行预测
        probability = model.predict_proba(features)[0, 1]  # 获取正类概率
        prediction = model.predict(features)[0]  # 获取预测类别

        # 渲染结果到网页
        return render_template(
            'result.html',
            prediction=int(prediction),
            probability=float(probability)
        )
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
