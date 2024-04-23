
import cv2
import numpy as np
from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array

# Load model from JSON file
json_file = open('fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load weights and them to model
model.load_weights('fer.h5')

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Check if Haar cascade classifier loaded successfully
if face_haar_cascade.empty():
    print("Error: Haar cascade classifier not loaded successfully.")
else:
    print("Haar cascade classifier loaded successfully.")

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.1, 6, minSize=(150, 150))

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        roi_gray = gray_img[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))  # Resize ROI to match model input size
        img_pixels = img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels = np.expand_dims(img_pixels, axis=-1)  # Add channel dimension
        img_pixels /= 255.0

        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions)

        emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
        predicted_emotion = emotions[max_index]

        # Classify into satisfied and not satisfied
        if predicted_emotion in ['neutral']:
            customer_status = "neutral"
        elif predicted_emotion in ['happiness','surprise']:
            customer_status = "satisfied"
        else:
            customer_status = "not satisfied"

        cv2.putText(img, f"Customer status: {customer_status}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    resized_img = cv2.resize(img, (1000, 700))
    cv2.imshow('Customer Satisfaction Detection', resized_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
