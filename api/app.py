from flask import Flask, request, jsonify
import requests
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Emotion Detector API',
        'status': 'running',
        'dependencies': 'Flask + Requests + NumPy',
        'endpoints': {
            '/health': 'GET - Health check',
            '/predict': 'POST - Predict emotion from image'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'API is running successfully',
        'numpy_version': np.__version__
    })

@app.route('/predict', methods=['POST'])
def predict_emotion():
    return jsonify({
        'error': 'Model not yet configured. Testing with NumPy.',
        'status': 'placeholder',
        'numpy_test': np.array([1, 2, 3]).tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
