import cv2
from keras.models import load_model
import numpy as np

# Load the model
model = load_model("best_emotion_model_v2.h5")  # or "emotion_model_final_v2.h5"

# Load Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to preprocess image for model
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 96, 96, 1)  # Match the model's input shape
    return feature / 255.0

# Start webcam
webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    ret, im = webcam.read()
    if not ret:
        continue
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    try:
        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            image = cv2.resize(image, (96, 96))  # Resize to (96, 96) to match training
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            cv2.putText(im, prediction_label, (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        
        cv2.imshow("Output", im)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    
    except cv2.error:
        pass

# Release resources
webcam.release()
cv2.destroyAllWindows()