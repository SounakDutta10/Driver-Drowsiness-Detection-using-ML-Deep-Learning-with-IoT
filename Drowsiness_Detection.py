import pickle
import cv2
import imutils
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
import winsound  # Use pygame.mixer for cross-platform support

# Load the DNN model
with open("models/dnn_model.pkl", "rb") as f:
    model = pickle.load(f)


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[1], mouth[7])  # P2 - P8 
    B = distance.euclidean(mouth[3], mouth[5])  # P4 - P6
    C = distance.euclidean(mouth[0], mouth[4])  # P1 - P5
    return (A + B) / (2.0 * C)

thresh_ear = 0.25
thresh_mar = 0.6
frame_check = 20

cap = cv2.VideoCapture(0)
flag = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use the DNN model for face detection
    faces = model.detect_faces(gray)
    
    for face in faces:
        shape = model.predict_landmarks(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"][0]:face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"][1]]
        rightEye = shape[face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"][0]:face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"][1]]
        mouth = shape[face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"][0]:face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"][1]]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(mouth)
        
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 0, 255), 1)
        
        if ear < thresh_ear:
            flag += 1
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                winsound.Beep(1000, 500)
        else:
            flag = 0
        
        if mar > thresh_mar:
            cv2.putText(frame, "Yawning!", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            winsound.Beep(1200, 500)
            
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
