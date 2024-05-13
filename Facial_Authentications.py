import cv2
import numpy as np

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load pre-trained recognizer model
recognizer.read('trained_model.yml')

# Function to perform face recognition
def recognize_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        # Predict face using recognizer
        label, confidence = recognizer.predict(roi_gray)
        
        # If confidence is less than 100, the face is recognized with some level of certainty
        if confidence < 100:
            cv2.putText(img, f'User {label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            cv2.putText(img, 'Unknown', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    return img

# Function to capture video from webcam and perform face recognition
def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = recognize_face(frame)
        cv2.imshow('Facial Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
