from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model("model.h5")
model.load_weights("modelWeights.weights.h5")

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
