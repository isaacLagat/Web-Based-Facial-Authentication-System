This code performs the following steps:

Loads the pre-trained face detection model (Haar Cascade) and LBPH face recognizer.
Uses the face detection model to detect faces in the webcam feed.
For each detected face, it crops the region of interest (ROI), converts it to grayscale, and passes it to the LBPH face recognizer for prediction.
If the recognizer predicts a face with confidence less than 100, it labels the face with the user ID and draws a green rectangle around it. Otherwise, it labels it as 'Unknown' and draws a red rectangle.
Displays the processed frame with recognized faces in real-time using OpenCV.
You'll need to train the LBPH face recognizer beforehand using a dataset of facial images corresponding to different users. After training, save the model as 'trained_model.yml' and place it in the same directory as the script
