from flask import Flask, request, jsonify
import joblib
import numpy as np

# 加载模型
model = joblib.load("bleeding_risk_model.pkl")

# 初始化 Flask 应用
app = Flask(__name__)

@app.route('/')
def home():
    return "Bleeding Risk Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    probability = model.predict_proba(features)[0, 1]
    prediction = model.predict(features)[0]
    return jsonify({
        "prediction": int(prediction),
        "probability": float(probability)
    })

if __name__ == '__main__':
    import os
    # 从 Render 的环境变量中获取端口
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

