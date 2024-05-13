import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained VGGFace model for deep feature extraction
vggface_model = load_model('vggface_model.h5')  # You need to download or train the model first

# Load pre-trained SVM model for traditional feature-based recognition
svm_model = SVC(kernel='linear', probability=True)  # You need to train the SVM model first

# Function to preprocess and extract features using VGGFace model
def extract_features(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = vggface_model.predict(img)
    return features

# Function to train SVM model (replace with your actual training code)
def train_svm_model():
    # Load training data and labels
    X_train = np.load('train_features.npy')
    y_train = np.load('train_labels.npy')
    svm_model.fit(X_train, y_train)

# Function to perform face recognition using both SVM and CNN
def recognize_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        # Extract features using CNN
        cnn_features = extract_features(img)
        # Predict using SVM
        svm_label = svm_model.predict(cnn_features)
        
        # If SVM predicts a label with high confidence, display the label and draw a green rectangle
        if svm_label[0] == 1:  # Example label for a recognized user
            cv2.putText(img, f'User {svm_label[0]}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            # Label as 'Unknown' and draw a red rectangle
            cv2.putText(img, 'Unknown', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    return img

# Function to capture video from webcam and perform face recognition
def main():
    # Load trained SVM model
    train_svm_model()
    
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
