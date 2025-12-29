import cv2
import mediapipe as mp
import math
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def fingers_up(hand, handedness):
    tips = [4, 8, 12, 16, 20]
    pip =  [3, 6, 10, 14, 18]
    res = []

    # index–pinky
    for i in range(1,5):
        tip = hand.landmark[tips[i]]
        pip_l = hand.landmark[pip[i]]
        res.append(tip.y < pip_l.y)

    # thumb
    tip = hand.landmark[tips[0]]
    ip = hand.landmark[pip[0]]

    # If flipped image, handedness is mirrored — test both & flip if needed
    if handedness == "Right":
        res.insert(0, tip.x < ip.x)
    else:
        res.insert(0, tip.x > ip.x)

    return res  # thumb, idx, mid, ring, pinky

def classify(hand, handedness):
    f = fingers_up(hand, handedness)
    thumb, idx, mid, ring, pinky = f

    wrist = hand.landmark[0]
    thumb_tip = hand.landmark[4]
    index_tip = hand.landmark[8]

    # Thumbs-Up
    if thumb and not(idx or mid or ring or pinky):
        if thumb_tip.y < wrist.y:
            return "Thumbs Up"
        if thumb_tip.y > wrist.y:
            return "Thumbs Down"

    if all(f):
        return "Open"

    if not any(f):
        return "Fist"

    if idx and mid and not ring and not pinky:
        return "Peace"

    # SINGLE FINGER DETECTION (separate for each finger)

    # Only thumb up
    if thumb and not(idx or mid or ring or pinky):
        return "THUMB_ONLY"

    # Only index up
    if idx and not(thumb or mid or ring or pinky):
        return "INDEX_ONLY"

    # Only middle up
    if mid and not(thumb or idx or ring or pinky):
        return "MIDDLE_ONLY"

    # Only ring up
    if ring and not(thumb or idx or mid or pinky):
        return "RING_ONLY"

    # Only pinky up
    if pinky and not(thumb or idx or mid or ring):
        return "PINKY_ONLY"

    return None

# GESTURE → REAL-WORLD TEXT MAPPING
GESTURE_TEXT = {
    # Core control gestures
    "Open": "READY",
    "Fist": "STOP",
    "Thumbs Up": "CONFIRM",
    "Thumbs Down": "REJECT",
    "Peace": "NEXT",

    # Single-finger gestures
    "INDEX_ONLY": "SELECT",
    "MIDDLE_ONLY": "MODE",
    "RING_ONLY": "OPTION",
    "PINKY_ONLY": "HELP",
    "THUMB_ONLY": "BACK",
}


# timing variables
stable_gesture = None
current_gesture = None
gesture_start_time = 0
STABLE_TIME = 0.5

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)

        gesture_this_frame = None

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                label = handedness.classification[0].label
                g = classify(hand_landmarks, label)
                if g:
                    gesture_this_frame = g

        # 0.5 SEC STABILIZATION LOGIC
        if gesture_this_frame != current_gesture:
            current_gesture = gesture_this_frame
            gesture_start_time = time.time()
            stable_gesture = None
        else:
            if current_gesture and (time.time() - gesture_start_time >= STABLE_TIME):
                stable_gesture = current_gesture

        # TEXT AT BOTTOM
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, h-40), (w, h), (0,0,0), -1)
        txt = GESTURE_TEXT.get(stable_gesture, "") if stable_gesture else ""
        cv2.putText(frame, txt, (10, h-10), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 2)

        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()