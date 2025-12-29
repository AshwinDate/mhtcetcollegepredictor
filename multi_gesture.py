# =========================
#   MULTI GESTURE DETECTOR
# =========================
# Works on Mediapipe Hands 0.10.x + OpenCV
# Supports gestures:
# Peace, Hang Loose, Loser, High Five, Talk to the Hand,
# You (Point), Good Job (Thumbs Up), Dislike (Thumbs Down),
# OK, A-hole (Middle Finger), Rock, Bang Bang (Finger Gun),
# Call Me, Fist (fallback)
# =========================

# -------- HIDE TFLITE / MEDIAPIPE WARNINGS ----------
# Put this at the very top of the file (line 1). No imports should appear before it.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # 0=all,1=INFO off,2=WARNING off,3=ERROR only

# optional: silencing absl logs if absl is present later
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass

# now other imports
import cv2
import mediapipe as mp
# ... rest of your code ...

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide INFO + WARNING logs
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except:
    pass

# ---------------- IMPORTS ----------------
import cv2
import mediapipe as mp
import math
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ---------------- CONSTANTS ----------------
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

INDEX_PIP = 6
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18
THUMB_IP = 3

INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17
THUMB_MCP = 2

# ------------ HELPER FUNCTIONS -------------
def vec(a, b):
    return (b[0] - a[0], b[1] - a[1])

def length(v):
    return math.hypot(v[0], v[1])

def angle(u, v):
    dotp = u[0]*v[0] + u[1]*v[1]
    lu = length(u)
    lv = length(v)
    if lu * lv == 0:
        return 180
    return math.degrees(math.acos(max(min(dotp / (lu * lv), 1), -1)))

def finger_extended(lm, mcp, pip, tip, threshold=60):
    a = vec(pip, mcp)
    b = vec(pip, tip)
    return angle(a, b) < threshold

def thumb_extended(lm, w, h):
    tip = lm[THUMB_TIP]
    ip = lm[THUMB_IP]
    wrist = lm[WRIST]
    right_hand = lm[INDEX_MCP][0] > wrist[0]

    dx = tip[0] - ip[0]
    dist = length(vec(tip, wrist)) / math.hypot(w, h)

    if dist < 0.03:
        return False
    return dx > 10 if right_hand else dx < -10

# ---------------- GESTURE DETECTOR ----------------
def detect(lm, w, h):
    idx = finger_extended(lm, INDEX_MCP, INDEX_PIP, INDEX_TIP)
    mid = finger_extended(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP)
    ring = finger_extended(lm, RING_MCP, RING_PIP, RING_TIP)
    pinky = finger_extended(lm, PINKY_MCP, PINKY_PIP, PINKY_TIP)
    thumb = thumb_extended(lm, w, h)

    diag = math.hypot(w, h)
    thumb_index_dist = length(vec(lm[THUMB_TIP], lm[INDEX_TIP])) / diag

    # ---- High Five / Open Palm ----
    if idx and mid and ring and pinky and thumb:
        return "High Five (Open Palm)"

    # ---- Peace / V Sign ----
    if idx and mid and not ring and not pinky:
        return "Peace"

    # ---- Hang Loose / Shaka ----
    if thumb and pinky and not idx and not mid and not ring:
        return "Hang Loose"

    # ---- Loser (L-shape: index up, thumb left/right) ----
    if idx and thumb and not mid and not ring and not pinky:
        # check right angle between index and thumb
        return "Loser"

    # ---- You / Point ----
    if idx and not mid and not ring and not pinky and not thumb:
        return "You (Point)"

    # ---- Thumbs Up / Down ----
    if thumb and not idx and not mid and not ring and not pinky:
        if lm[THUMB_TIP][1] < lm[WRIST][1]:
            return "Good Job (Thumbs Up)"
        else:
            return "Dislike (Thumbs Down)"

    # ---- OK ----
    if thumb_index_dist < 0.03:
        return "OK"

    # ---- Middle Finger ----
    if mid and not idx and not ring and not pinky:
        return "A-hole (Middle Finger)"

    # ---- Rock / Horns ----
    if idx and pinky and not mid and not ring:
        return "Rock"

    # ---- Finger Gun / Bang Bang ----
    if idx and thumb and not mid and not ring and not pinky and thumb_index_dist > 0.04:
        return "Bang Bang"

    # ---- Call Me ----
    if thumb and pinky and (idx or mid):
        return "Call Me"

    # ---- Talk to the Hand ----
    if idx and mid and ring and pinky and not thumb:
        return "Talk to the Hand"

    # ---- Fist ----
    if not idx and not mid and not ring and not pinky and not thumb:
        return "Fist"

    return "Unknown"

# ---------------- MAIN PROGRAM ----------------
def main():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        prev = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img.flags.writeable = False
            res = hands.process(img)
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            gesture = "None"

            if res.multi_hand_landmarks:
                for hand in res.multi_hand_landmarks:
                    lm = [(int(p.x * w), int(p.y * h)) for p in hand.landmark]
                    gesture = detect(lm, w, h)
                    mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

            cv2.putText(img, f"Gesture: {gesture}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            # FPS
            now = time.time()
            fps = int(1 / (now - prev)) if prev else 0
            prev = now
            cv2.putText(img, f"FPS: {fps}", (10, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            cv2.imshow("Multi Gesture Detector", img)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
