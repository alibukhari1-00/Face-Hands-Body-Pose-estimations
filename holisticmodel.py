# Holistic Tracking using Mediapipe
import mediapipe as mp
import cv2
import pandas as pd

# Initialize Mediapipe Holistic module
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
draw_spec = mp_draw.DrawingSpec(thickness=2, circle_radius=2)

# Open webcam
feed = cv2.VideoCapture(0)

# Lists to store coordinates
face_coords_list = []
left_hand_coords_list = []
right_hand_coords_list = []
pose_coords_list = []

with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as holistic:

    while True:
        ret, frame = feed.read()
        if not ret:
            print("Feed Not Found !!!!")
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)
        height, width, _ = frame.shape

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        # ====== Face landmarks ======
        if results.face_landmarks:
            mp_draw.draw_landmarks(
                frame,
                results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=draw_spec
            )
            face_coords = [[int(lm.x * width), int(lm.y * height)] for lm in results.face_landmarks.landmark]
            face_coords_list.append(face_coords)

        # ====== Left hand landmarks ======
        if results.left_hand_landmarks:
            mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=draw_spec)
            left_hand_coords = [[int(lm.x * width), int(lm.y * height)] for lm in results.left_hand_landmarks.landmark]
            left_hand_coords_list.append(left_hand_coords)

        # ====== Right hand landmarks ======
        if results.right_hand_landmarks:
            mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=draw_spec)
            right_hand_coords = [[int(lm.x * width), int(lm.y * height)] for lm in results.right_hand_landmarks.landmark]
            right_hand_coords_list.append(right_hand_coords)

        # ====== Pose landmarks ======
        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=draw_spec)
            pose_coords = [[int(lm.x * width), int(lm.y * height)] for lm in results.pose_landmarks.landmark]
            pose_coords_list.append(pose_coords)

        cv2.imshow("Holistic Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

feed.release()
cv2.destroyAllWindows()

# Save all coordinates to CSV
pd.DataFrame(face_coords_list).to_csv("holistic_face.csv", index=False)
pd.DataFrame(left_hand_coords_list).to_csv("holistic_left_hand.csv", index=False)
pd.DataFrame(right_hand_coords_list).to_csv("holistic_right_hand.csv", index=False)
pd.DataFrame(pose_coords_list).to_csv("holistic_pose.csv", index=False)

print("Face, hands, and pose landmarks saved to CSV files successfully!")
