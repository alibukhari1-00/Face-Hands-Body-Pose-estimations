import cv2
import mediapipe as mp
import pandas as pd
import math

# Initialize BlazePose
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
draw_spec = mp_draw.DrawingSpec(thickness=2, circle_radius=3)

# Open webcam
cap = cv2.VideoCapture(0)

# Function to calculate distance between two points
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# Store measurements
measurements_list = []

# ---- Set your actual height in cm ----
actual_height_cm = 170  # replace with your real height

with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as pose:

    scale_factor = None  # cm per pixel

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Feed Not Found!")
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=draw_spec,
                connection_drawing_spec=mp_draw.DrawingSpec(color=(0,255,0), thickness=2)
            )

            lm = results.pose_landmarks.landmark
            landmarks = [[int(l.x*width), int(l.y*height)] for l in lm]

            # Compute scale factor using head-top to left-ankle distance (pixel)
            if scale_factor is None:
                pixel_height = distance(landmarks[0], landmarks[27])
                scale_factor = actual_height_cm / pixel_height


            # Measurements in pixels
            shoulder_width_px = distance(landmarks[11], landmarks[12])
            torso_length_px = distance(landmarks[11], landmarks[23])
            left_arm_px = distance(landmarks[11], landmarks[13]) + distance(landmarks[13], landmarks[15])
            right_arm_px = distance(landmarks[12], landmarks[14]) + distance(landmarks[14], landmarks[16])
            left_leg_px = distance(landmarks[23], landmarks[25]) + distance(landmarks[25], landmarks[27])
            right_leg_px = distance(landmarks[24], landmarks[26]) + distance(landmarks[26], landmarks[28])
            height_px = distance(landmarks[0], landmarks[27])

            # Convert to cm
            shoulder_width_cm = shoulder_width_px * scale_factor
            torso_length_cm = torso_length_px * scale_factor
            left_arm_cm = left_arm_px * scale_factor
            right_arm_cm = right_arm_px * scale_factor
            left_leg_cm = left_leg_px * scale_factor
            right_leg_cm = right_leg_px * scale_factor
            height_cm = height_px * scale_factor

            # Draw on frame
            cv2.putText(frame, f'Shoulder: {shoulder_width_cm:.1f}cm', (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            cv2.putText(frame, f'Torso: {torso_length_cm:.1f}cm', (10,60), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            cv2.putText(frame, f'L Arm: {left_arm_cm:.1f}cm', (10,90), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            cv2.putText(frame, f'R Arm: {right_arm_cm:.1f}cm', (10,120), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            cv2.putText(frame, f'L Leg: {left_leg_cm:.1f}cm', (10,150), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            cv2.putText(frame, f'R Leg: {right_leg_cm:.1f}cm', (10,180), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            cv2.putText(frame, f'Height: {height_cm:.1f}cm', (10,210), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

            # Save to list
            measurements_list.append([shoulder_width_cm, torso_length_cm, left_arm_cm, right_arm_cm, left_leg_cm, right_leg_cm, height_cm])

        cv2.imshow("BlazePose Body Measurements (cm)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

# Save measurements to CSV
df = pd.DataFrame(measurements_list, columns=['Shoulder_cm','Torso_cm','Left_Arm_cm','Right_Arm_cm','Left_Leg_cm','Right_Leg_cm','Height_cm'])
df.to_csv("body_measurements_cm.csv", index=False)
print("Body measurements saved to body_measurements_cm.csv")

# Print last frame measurements
if measurements_list:
    last = measurements_list[-1]
    print("\nLast Frame Measurements in cm:")
    print(f"Shoulder: {last[0]:.1f} cm")
    print(f"Torso: {last[1]:.1f} cm")
    print(f"Left Arm: {last[2]:.1f} cm")
    print(f"Right Arm: {last[3]:.1f} cm")
    print(f"Left Leg: {last[4]:.1f} cm")
    print(f"Right Leg: {last[5]:.1f} cm")
    print(f"Height: {last[6]:.1f} cm")
