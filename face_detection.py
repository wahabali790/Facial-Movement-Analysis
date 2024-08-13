import cv2
import mediapipe as mp

# Initiate holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)

        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Check if the person is in the right position
        if results.face_landmarks and results.pose_landmarks:
            nose_landmark = results.face_landmarks.landmark[1]  # Assuming 1 is the index for the nose tip
            #print("nose_landmarks", nose_landmark)
            left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
            #print("left sholder", left_shoulder)
            right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
            #print("right shoulder", right_shoulder)
            lip_bottom = results.face_landmarks.landmark[152]  # Index for LOWER_LIP
            #print('lip_bottom', lip_bottom)

            # Adjust these thresholds according to your needs
            nose_threshold = 0.05
            shoulder_threshold = 0.05
            lip_threshold = 0.05

            # Calculate average shoulder position
            avg_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2

            # Check nose, lip, and shoulder positions
            if abs(nose_landmark.x - avg_shoulder_x) < nose_threshold and abs(lip_bottom.x - avg_shoulder_x) < lip_threshold:
                cv2.putText(image, "Your face position is good", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            2, cv2.LINE_AA)  # FONT_HERSHEY_SIMPLEX is font type ,1 for size of text and 2 for thinkness
            else:
                cv2.putText(image, "Adjust your face position", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            2, cv2.LINE_AA)
        else:
            cv2.putText(image, "place your face in front of camera", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                        2, cv2.LINE_AA)

        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
