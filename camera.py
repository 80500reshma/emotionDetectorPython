import cv2
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("model.h5")
model.load_weights("modelWeights.weights.h5")

label_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48,48))
    face_input = np.expand_dims(face, axis=0).reshape(1,48,48,1) / 255.0

    prediction = model.predict(face_input)
    emotion_index = np.argmax(prediction[0])
    emotion = label_dict[emotion_index]

    cv2.putText(frame, emotion, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
