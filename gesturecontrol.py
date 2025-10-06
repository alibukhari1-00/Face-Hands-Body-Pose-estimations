import cv2
import mediapipe as mp
import pyautogui
import time
import math

# -------------------------
# Setup MediaPipe Hands
# -------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# -------------------------
# Setup Camera
# -------------------------
cap = cv2.VideoCapture(0)
prev_gesture = None
gesture_cooldown = 1  # seconds
last_action_time = time.time()

# -------------------------
# Helper Functions
# -------------------------
def distance(point1, point2):
    return math.hypot(point2.x - point1.x, point2.y - point1.y)

def detect_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    # Gesture 1: Thumbs Up → Volume Up
    if thumb_tip.y < thumb_ip.y and distance(index_tip, middle_tip) > 0.05:
        return "VOLUME_UP"

    # Gesture 2: Thumbs Down → Volume Down
    if thumb_tip.y > thumb_ip.y and distance(index_tip, middle_tip) > 0.05:
        return "VOLUME_DOWN"

    # Gesture 3: Index Swipe Right → Next Track
    if index_tip.x - index_mcp.x > 0.2:
        return "NEXT_TRACK"

    # Gesture 4: Index Swipe Left → Previous Track
    if index_mcp.x - index_tip.x > 0.2:
        return "PREV_TRACK"

    return None

# -------------------------
# Main Loop
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror the frame
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    gesture = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detect_gesture(hand_landmarks)

    # Perform actions with cooldown
    current_time = time.time()
    if gesture and gesture != prev_gesture and current_time - last_action_time > gesture_cooldown:
        if gesture == "VOLUME_UP":
            pyautogui.press("volumeup")
            print("Volume Up")
        elif gesture == "VOLUME_DOWN":
            pyautogui.press("volumedown")
            print("Volume Down")
        elif gesture == "NEXT_TRACK":
            pyautogui.press("nexttrack")
            print("Next Track")
        elif gesture == "PREV_TRACK":
            pyautogui.press("prevtrack")
            print("Previous Track")

        prev_gesture = gesture
        last_action_time = current_time

    # Display gesture text on screen
    if gesture:
        cv2.putText(frame, f'Gesture: {gesture}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
