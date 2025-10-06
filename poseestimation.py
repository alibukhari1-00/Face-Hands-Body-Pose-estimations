# Pose Estimation using Mediapipe
import mediapipe as mp
import cv2
import pandas as pd

# Initialize Mediapipe Pose module
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
draw_spec = mp_draw.DrawingSpec(thickness=2, circle_radius=3)

# Open webcam
feed = cv2.VideoCapture(0)
pose_landmarks_list = []  # Store landmarks for each frame

with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as pose:

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
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Draw landmarks and connections
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=draw_spec,
                connection_drawing_spec=mp_draw.DrawingSpec(color=(0,255,0), thickness=2)
            )

            # Save landmarks coordinates (pixel values)
            pose_coords = []
            for lm in results.pose_landmarks.landmark:
                x, y = int(lm.x * width), int(lm.y * height)
                pose_coords.append([x, y])
            pose_landmarks_list.append(pose_coords)

        cv2.imshow("Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

feed.release()
cv2.destroyAllWindows()

# Save pose landmarks to CSV
df = pd.DataFrame(pose_landmarks_list)
df.to_csv("pose_landmarks.csv", index=False)
print("Pose landmarks saved to pose_landmarks.csv")
