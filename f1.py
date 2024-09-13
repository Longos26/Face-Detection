import cv2
from random import randrange
from fer import FER

# Load the pre-trained face detection model
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capture video from the webcam
webcam = cv2.VideoCapture(0)

# Initialize the emotion detector
emotion_detector = FER()

# Loop through frames from the video stream
while True:
    # Read a frame from the video stream
    successful_frame_read, frame = webcam.read()

    # If frame reading is unsuccessful, break the loop
    if not successful_frame_read:
        break

    # Convert the frame to grayscale for face detection
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = trained_face_data.detectMultiScale(grayscaled_frame, scaleFactor=1.1, minNeighbors=5)

    # Loop through all detected faces
    for (x, y, w, h) in faces:
        # Extract the face region from the frame for emotion detection
        face_roi = frame[y:y+h, x:x+w]

        # Use the emotion detector on the face region
        emotion_predictions = emotion_detector.detect_emotions(face_roi)

        # If any emotion is detected
        if emotion_predictions:
            # Get the most likely emotion
            dominant_emotion = emotion_predictions[0]['emotions']
            emotion_label = max(dominant_emotion, key=dominant_emotion.get)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 3)

            # Display the detected emotion above the face
            cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame with detected faces and emotions
    cv2.imshow('Romar Face & Emotion Detection', frame)

    # Wait for 1 millisecond and check if the user wants to quit (by pressing 'q')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()

print("Face and emotion detection complete!")
