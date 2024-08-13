from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

app = Flask(__name__)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def generate_frames():
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.face_landmarks and results.pose_landmarks:
                nose_landmark = results.face_landmarks.landmark[1]  
                left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
                lip_bottom = results.face_landmarks.landmark[152]  

                nose_threshold = 0.05
                shoulder_threshold = 0.05
                lip_threshold = 0.05

                avg_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2

                if abs(nose_landmark.x - avg_shoulder_x) < nose_threshold and abs(lip_bottom.x - avg_shoulder_x) < lip_threshold:
                    cv2.putText(image, "Your face position is good", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                2, cv2.LINE_AA)  
                else:
                    cv2.putText(image, "Adjust your face position", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                2, cv2.LINE_AA)
            else:
                cv2.putText(image, "Place your face in front of camera", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/face_adjustment')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
