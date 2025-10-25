from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
import requests

# URLs to your GitHub Release files
MODEL_URL = "https://github.com/80500reshma/emotionDetectorPython/releases/download/v1.0/model.h5"
WEIGHTS_URL = "https://github.com/80500reshma/emotionDetectorPython/releases/download/v1.0/modelWeights.weights.h5"

# Local paths
MODEL_PATH = "model.h5"
WEIGHTS_PATH = "modelWeights.weights.h5"

# Download files if they don't exist
def download_file(url, path):
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        r = requests.get(url, allow_redirects=True)
        with open(path, 'wb') as f:
            f.write(r.content)
        print(f"{path} downloaded!")

download_file(MODEL_URL, MODEL_PATH)
download_file(WEIGHTS_URL, WEIGHTS_PATH)

# Load model and weights
model = tf.keras.models.load_model(MODEL_PATH)
model.load_weights(WEIGHTS_PATH)

label_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_emotion():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    img_bytes = file.read()
    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)

    # Preprocess image
    img = cv2.resize(img, (48,48))
    img = np.expand_dims(img, axis=0).reshape(1,48,48,1) / 255.0

    result = model.predict(img)
    emotion_index = np.argmax(result[0])
    emotion = label_dict[emotion_index]

    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
