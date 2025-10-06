# Face detection using mediapipe
import mediapipe as mp
import cv2

mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

feed = cv2.VideoCapture(0)

with mp_face.FaceDetection(min_detection_confidence=0.7) as fd:
    while True:
        ret, frame = feed.read()
        frame=cv2.flip(frame,1)
        frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)  # alpha=contrast, beta=brightness
        if not ret:
            print("Feed Not Found !!!!")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = fd.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                mp_draw.draw_detection(frame, detection)
            print("Confidence Score is ",detection.score)

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

feed.release()
cv2.destroyAllWindows()
