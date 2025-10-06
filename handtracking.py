# Hand Tracking using Mediapipe
import mediapipe as mp
import cv2
import pandas as pd

# Initialize mediapipe hand module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
draw_spec = mp_draw.DrawingSpec(thickness=2, circle_radius=3)

# Open webcam
feed = cv2.VideoCapture(0)
hand_landmarks_list = []  # Store landmarks for each frame

with mp_hands.Hands(
        max_num_hands=2,  # Track up to 2 hands
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

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
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks and connections
                mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=draw_spec
                )

                # Save landmarks coordinates (pixel values)
                hand_coords = []
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * width), int(lm.y * height)
                    hand_coords.append([x, y])
                hand_landmarks_list.append(hand_coords)

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

feed.release()
cv2.destroyAllWindows()

# Save hand landmarks to CSV
df = pd.DataFrame(hand_landmarks_list)
df.to_csv("hand_landmarks.csv", index=False)
print("Hand landmarks saved to hand_landmarks.csv")
