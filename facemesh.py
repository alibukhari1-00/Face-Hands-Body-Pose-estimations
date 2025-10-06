# Face Mesh using Mediapipe
import mediapipe as mp
import cv2
import pandas as pd

# Initialize mediapipe face mesh and drawing utils
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1)

# Open webcam
feed = cv2.VideoCapture(0)
landmarks_list = []  # Store landmarks

with mp_face_mesh.FaceMesh(
        max_num_faces=1,  # Number of faces to track
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as fm:

    while True:
        ret, frame = feed.read()
        if not ret:
            print("Feed Not Found !!!!")
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)
        height, width, _ = frame.shape

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = fm.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw landmarks
                mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=draw_spec,
                    connection_drawing_spec=mp_draw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
                )

                # Save landmarks coordinates (pixel values)
                face_coords = []
                for lm in face_landmarks.landmark:
                    x, y = int(lm.x * width), int(lm.y * height)
                    face_coords.append([x, y])
                landmarks_list.append(face_coords)

        cv2.imshow("Face Mesh", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

feed.release()
cv2.destroyAllWindows()

# Save landmarks to CSV
df = pd.DataFrame(landmarks_list)
df.to_csv("face_mesh_landmarks.csv", index=False)
print("Face landmarks saved to face_mesh_landmarks.csv")
