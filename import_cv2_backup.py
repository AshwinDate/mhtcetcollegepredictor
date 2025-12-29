"""
hand_sign_with_icons_and_subtitles.py

Contour-based hand-sign detector (Python 3.13) with icon overlay and subtitles.

- Auto-slices spritesheet if ./images.jpg exists (6 columns x 5 rows default).
- Loads icons from ./icons/
- Detects finger pattern (thumb,index,middle,ring,pinky) -> matches an icon index + subtitle
- Overlays icon and writes subtitle to subtitle.txt

Install:
    pip install opencv-python numpy

Run:
    python hand_sign_with_icons_and_subtitles.py
"""
import cv2
import numpy as np
import math
import os
from collections import deque

# ---------------- CONFIG ----------------
CAM_INDEX = 0
WINDOW = "Hand Signs with Icons"
MIN_CONTOUR_AREA = 3000
HISTORY = 10
ICONS_DIR = "icons"
SPRITESHEET = "images.jpg"   # place your spritesheet next to the script or set full path
SPRITESHEET_COLS = 6
SPRITESHEET_ROWS = 5
WRITE_SUBTITLE_FILE = "subtitle.txt"

# Detection tuning
DEFECT_DEPTH_RATIO = 0.02
DEFECT_ANGLE_THRESHOLD = 90

# Map simple finger patterns to (icon_index, subtitle)
# Pattern is tuple of (thumb, index, middle, ring, pinky) -> 1 = extended, 0 = folded
# Edit these mappings to match icon indices produced by your spritesheet slicing.
PATTERN_TO_ICON = {
    (0,0,0,0,0): (0, "FIST / STOP"),
    (0,1,0,0,0): (1, "POINTING"),
    (0,1,1,0,0): (2, "PEACE"),
    (0,1,1,1,0): (3, "THREE"),
    (0,1,1,1,1): (4, "FOUR"),
    (1,1,1,1,1): (5, "OPEN HAND"),
    (1,0,0,0,0): (6, "THUMBS UP (approx)"),
    # You can add more mappings here. Icon indices correspond to icon_#.png
}

# (Optional) A fallback icon index when pattern not mapped (set None to not show)
FALLBACK_ICON_INDEX = None

# ---------------- utility: spritesheet slice & icons load ----------------
def ensure_icons_dir():
    os.makedirs(ICONS_DIR, exist_ok=True)

def autoslice_spritesheet(path, out_dir, cols=SPRITESHEET_COLS, rows=SPRITESHEET_ROWS):
    if not os.path.exists(path):
        return 0
    img = cv2.imread(path)
    if img is None:
        return 0
    h, w = img.shape[:2]
    cw = w // cols
    ch = h // rows
    count = 0
    for r in range(rows):
        for c in range(cols):
            x = c * cw
            y = r * ch
            cell = img[y:y+ch, x:x+cw]
            fname = os.path.join(out_dir, f"icon_{count}.png")
            cv2.imwrite(fname, cell)
            count += 1
    return count

def load_icons(max_index_guess=30):
    icons = {}
    # try to load icons icon_0..icon_{max_index_guess-1}
    for i in range(max_index_guess):
        path = os.path.join(ICONS_DIR, f"icon_{i}.png")
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            icons[i] = img
    return icons

# ---------------- skin mask & geometry ----------------
def combined_skin_mask(frame):
    # YCrCb
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    lower_y = np.array([0,133,77], dtype=np.uint8)
    upper_y = np.array([255,173,127], dtype=np.uint8)
    m1 = cv2.inRange(ycrcb, lower_y, upper_y)
    # HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_h = np.array([0,30,60], dtype=np.uint8)
    upper_h = np.array([25,200,255], dtype=np.uint8)
    m2 = cv2.inRange(hsv, lower_h, upper_h)
    mask = cv2.bitwise_or(m1, m2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    return mask

def largest_contour(contours):
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def wrist_estimate(contour, take_n=40):
    pts = contour.reshape(-1,2)
    idx = np.argsort(pts[:,1])[::-1]
    bottom = pts[idx[:take_n]]
    return (int(np.mean(bottom[:,0])), int(np.mean(bottom[:,1])))

def convexity_defect_fingertips(contour):
    pts = contour.reshape(-1,2)
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return []
    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return []
    bbox_h = cv2.boundingRect(contour)[3] or 1
    depth_thresh = max(10, DEFECT_DEPTH_RATIO * bbox_h)
    cand = []
    for i in range(defects.shape[0]):
        s,e,f,depth = defects[i,0]
        depth_real = depth / 256.0
        start = tuple(map(int, pts[s]))
        end = tuple(map(int, pts[e]))
        far = tuple(map(int, pts[f]))
        # angle at far
        a = np.array(start, dtype=np.float32)
        b = np.array(far, dtype=np.float32)
        c = np.array(end, dtype=np.float32)
        denom = (np.linalg.norm(a-b) * np.linalg.norm(c-b))
        if denom == 0:
            continue
        cosang = float(np.dot(a-b, c-b) / denom)
        cosang = max(-1.0, min(1.0, cosang))
        angle = math.degrees(math.acos(cosang))
        if depth_real > depth_thresh and angle < DEFECT_ANGLE_THRESHOLD:
            cand.append(start); cand.append(end)
    if not cand:
        return []
    cand = list(set(cand))
    cand = sorted(cand, key=lambda p: p[1])  # prefer higher points
    unique = []
    for p in cand:
        if not any(math.hypot(p[0]-q[0], p[1]-q[1]) < 30 for q in unique):
            unique.append(p)
    unique = sorted(unique, key=lambda p: p[0])  # left->right
    return unique[:5]

def map_tips_to_five(tips, wrist, centroid):
    # order left->right
    if not tips:
        return [None]*5
    tips_sorted = sorted(tips, key=lambda p: p[0])
    orientation = 'right' if centroid[0] < wrist[0] else 'left'
    if orientation == 'left':
        tips_sorted = list(reversed(tips_sorted))
    while len(tips_sorted) < 5:
        tips_sorted.append(None)
    return tips_sorted[:5]

# ---------------- subtitle & icon overlay helpers ----------------
def draw_subtitle(frame, text):
    h,w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.9, w/900)
    thickness = max(2, int(scale*2))
    (tw,th), _ = cv2.getTextSize(text, font, scale, thickness)
    padx = int(0.03*w); pady = int(0.02*h)
    box_w = tw + 2*padx; box_h = th + 2*pady
    x = (w - box_w)//2; y = h - box_h - 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (x,y), (x+box_w, y+box_h), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    tx = x + padx; ty = y + pady + th
    # outline + text
    cv2.putText(frame, text, (tx,ty), font, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(frame, text, (tx,ty), font, scale, (255,255,255), thickness, cv2.LINE_AA)

def overlay_icon(frame, icon, size=140, margin=20):
    if icon is None:
        return frame
    h,w = frame.shape[:2]
    icon_r = cv2.resize(icon, (size,size), interpolation=cv2.INTER_AREA)
    x = w - size - margin; y = margin
    if icon_r.ndim == 2:
        icon_r = cv2.cvtColor(icon_r, cv2.COLOR_GRAY2BGR)
    if icon_r.shape[2] == 4:
        alpha = icon_r[:,:,3].astype(np.float32)/255.0
        for c in range(3):
            frame[y:y+size, x:x+size, c] = (alpha*icon_r[:,:,c] + (1-alpha)*frame[y:y+size, x:x+size, c])
    else:
        frame[y:y+size, x:x+size] = icon_r[:,:,:3]
    return frame

# ---------------- main ----------------
def main():
    ensure_icons_dir = lambda : os.makedirs(ICONS_DIR, exist_ok=True)
    ensure_icons_dir()
    # autoslice if icons folder empty and spritesheet present
    if not os.listdir(ICONS_DIR) and os.path.exists(SPRITESHEET):
        autoslice_spritesheet(SPRITESHEET, ICONS_DIR, cols=SPRITESHEET_COLS, rows=SPRITESHEET_ROWS)
    icons = load_icons(max_index_guess=SPRITESHEET_COLS*SPRITESHEET_ROWS)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    hist = deque(maxlen=HISTORY)
    last_stable = ""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h,w = frame.shape[:2]
        mask = combined_skin_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cont = largest_contour(contours) if contours else None

        subtitle = "No hand"
        icon_to_show = None

        if cont is not None and cv2.contourArea(cont) > MIN_CONTOUR_AREA:
            try:
                approx = cv2.approxPolyDP(cont, 0.003*cv2.arcLength(cont, True), True)
                centroid = tuple(map(int, np.mean(approx.reshape(-1,2), axis=0)))
                wrist = wrist_estimate(approx)
                tips = convexity_defect_fingertips(approx)
                tips5 = map_tips_to_five(tips, wrist, centroid)

                # decide extended by comparing tip Y with a proximal joint along contour
                finger_states = []
                for tip in tips5:
                    if tip is None:
                        finger_states.append(0); continue
                    pts = approx.reshape(-1,2).astype(np.int32)
                    dists = np.sum((pts - np.array(tip))**2, axis=1)
                    idx = int(np.argmin(dists)); n = len(pts)
                    jprox = tuple(map(int, pts[(idx+30)%n]))
                    state = 1 if tip[1] < jprox[1] - 25 else 0
                    finger_states.append(state)

                pattern = tuple(finger_states)
                # lookup pattern in mapping
                if pattern in PATTERN_TO_ICON:
                    icon_idx, label = PATTERN_TO_ICON[pattern]
                    icon_to_show = icons.get(icon_idx)
                    subtitle = label
                else:
                    subtitle = f"{sum(pattern)} fingers"
                    if FALLBACK_ICON_INDEX is not None:
                        icon_to_show = icons.get(FALLBACK_ICON_INDEX)

                # draw skeleton points for debugging
                cv2.drawContours(frame, [approx], -1, (0,255,0), 1)
                hull = cv2.convexHull(approx)
                cv2.drawContours(frame, [hull], -1, (255,0,0), 1)
                # draw wrist and tips
                cv2.circle(frame, wrist, 6, (255,255,255), -1)
                for t in tips:
                    cv2.circle(frame, t, 6, (0,0,255), -1)

                # small text pattern
                cv2.putText(frame, f"Pattern: {pattern}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            except Exception as e:
                subtitle = "Error detecting"

        hist.append(subtitle)
        try:
            stable = max(set(hist), key=hist.count)
        except Exception:
            stable = subtitle

        if stable != last_stable:
            try:
                with open(WRITE_SUBTITLE_FILE, "w", encoding="utf-8") as f:
                    f.write(stable)
            except Exception:
                pass
            last_stable = stable

        # render overlay
        draw_subtitle(frame, stable)
        if icon_to_show is not None:
            frame = overlay_icon(frame, icon_to_show, size=min(160, w//6))

        # mask preview
        try:
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            preview = cv2.resize(mask_bgr, (int(w*0.25), int(h*0.25)))
            ph, pw = preview.shape[:2]
            frame[0:ph, 0:pw] = preview
        except Exception:
            pass

        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('r'):  # reload icons
            icons = load_icons(max_index_guess=SPRITESHEET_COLS*SPRITESHEET_ROWS)
            print("Reloaded icons")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
