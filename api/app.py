from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Emotion Detector API',
        'status': 'running',
        'endpoints': {
            '/health': 'GET - Health check',
            '/predict': 'POST - Predict emotion from image'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'API is running successfully'
    })

@app.route('/predict', methods=['POST'])
def predict_emotion():
    return jsonify({
        'error': 'Model not yet configured. This is a test deployment.',
        'status': 'placeholder'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
