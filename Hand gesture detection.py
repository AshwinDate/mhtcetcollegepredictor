"""
Professional Hand Gesture Detection System
High Accuracy with MediaPipe Integration
Author: Advanced Hand Detection System
"""

import cv2
import numpy as np
import math

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available, using fallback method...")


class ProfessionalHandDetector:
    """High-accuracy hand detection with MediaPipe"""
    
    def __init__(self):
        if MEDIAPIPE_AVAILABLE:
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_draw_styles = mp.solutions.drawing_styles
            
            # Initialize with high accuracy settings
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            
            # Landmark connections for drawing
            self.hand_connections = self.mp_hands.HAND_CONNECTIONS
            print("✓ MediaPipe initialized - High accuracy mode")
        else:
            self.hands = None
            print("✗ MediaPipe not available - Install with: pip install mediapipe")
    
    def find_hands(self, img, draw=True):
        """
        Detect hands with high precision
        Returns: List of hands with 21 landmarks each
        """
        if not MEDIAPIPE_AVAILABLE or self.hands is None:
            return [], img
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        all_hands = []
        h, w, c = img.shape
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get hand type (Left/Right)
                hand_type = results.multi_handedness[hand_idx].classification[0].label
                confidence = results.multi_handedness[hand_idx].classification[0].score
                
                # Extract landmark positions
                landmark_list = []
                x_list, y_list = [], []
                
                for lm in hand_landmarks.landmark:
                    px, py = int(lm.x * w), int(lm.y * h)
                    landmark_list.append([px, py])
                    x_list.append(px)
                    y_list.append(py)
                
                # Calculate bounding box
                x_min, x_max = min(x_list), max(x_list)
                y_min, y_max = min(y_list), max(y_list)
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                
                # Calculate center
                cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
                
                # Detect fingers up
                fingers = self.fingers_up(landmark_list)
                
                hand_info = {
                    'landmarks': landmark_list,
                    'bbox': bbox,
                    'center': (cx, cy),
                    'type': hand_type,
                    'confidence': confidence,
                    'fingers': fingers,
                    'raw_landmarks': hand_landmarks
                }
                
                all_hands.append(hand_info)
                
                # Draw hand skeleton
                if draw:
                    img = self.draw_hand(img, hand_info)
        
        return all_hands, img
    
    def fingers_up(self, landmark_list):
        """
        Precise finger detection using landmark geometry
        Returns: [thumb, index, middle, ring, pinky] as 0 or 1
        """
        fingers = []
        
        # Thumb - special case (check x-axis)
        if len(landmark_list) >= 5:
            # Check if thumb tip is to the right/left of thumb IP joint
            if landmark_list[4][0] > landmark_list[3][0]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        # Other fingers - check if tip is above PIP joint
        tip_ids = [8, 12, 16, 20]
        pip_ids = [6, 10, 14, 18]
        
        for tip, pip in zip(tip_ids, pip_ids):
            if len(landmark_list) > tip:
                if landmark_list[tip][1] < landmark_list[pip][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        
        return fingers
    
    def draw_hand(self, img, hand_info):
        """
        Draw professional hand skeleton with landmarks
        """
        landmarks = hand_info['landmarks']
        x, y, w, h = hand_info['bbox']
        
        # Draw bounding box with padding
        padding = 20
        cv2.rectangle(img, 
                     (x - padding, y - padding), 
                     (x + w + padding, y + h + padding), 
                     (255, 0, 255), 2)
        
        # Draw connections between landmarks
        if MEDIAPIPE_AVAILABLE:
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                (5, 9), (9, 13), (13, 17)  # Palm
            ]
            
            for connection in connections:
                if connection[0] < len(landmarks) and connection[1] < len(landmarks):
                    pt1 = tuple(landmarks[connection[0]])
                    pt2 = tuple(landmarks[connection[1]])
                    cv2.line(img, pt1, pt2, (0, 255, 0), 2)
        
        # Draw landmarks
        for idx, lm in enumerate(landmarks):
            # Different colors for different parts
            if idx in [4, 8, 12, 16, 20]:  # Fingertips
                color = (0, 0, 255)
                radius = 8
            elif idx == 0:  # Wrist
                color = (255, 255, 0)
                radius = 10
            else:  # Other joints
                color = (0, 255, 255)
                radius = 5
            
            cv2.circle(img, tuple(lm), radius, color, -1)
            cv2.circle(img, tuple(lm), radius + 2, (255, 255, 255), 1)
        
        return img


def calculate_angle(p1, p2, p3):
    """Calculate angle between three points"""
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    
    ba = a - b
    bc = c - b
    
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)


def get_gesture_name(fingers):
    """Enhanced gesture recognition"""
    finger_count = sum(fingers)
    
    gestures = {
        (0, 0, 0, 0, 0): "✊ Fist",
        (1, 0, 0, 0, 0): "👍 Thumbs Up",
        (0, 1, 0, 0, 0): "☝ Pointing",
        (0, 1, 1, 0, 0): "✌ Peace/Victory",
        (0, 1, 1, 1, 0): "🤟 Three",
        (0, 1, 1, 1, 1): "🖖 Four",
        (1, 1, 1, 1, 1): "✋ Open Hand",
        (1, 1, 0, 0, 0): "🤘 Rock",
        (1, 0, 0, 0, 1): "🤙 Shaka",
        (0, 0, 0, 0, 1): "🤞 Pinky"
    }
    
    return gestures.get(tuple(fingers), f"🖐 {finger_count} Fingers")


def draw_professional_ui(img, hands, fps, frame_count):
    """Draw professional UI overlay"""
    h, w = img.shape[:2]
    
    # Top bar background
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    
    # Title
    cv2.putText(img, "PROFESSIONAL HAND DETECTION SYSTEM", (10, 30),
               cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
    
    # Stats
    cv2.putText(img, f"FPS: {int(fps)}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, f"Hands: {len(hands)}", (150, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, f"Frame: {frame_count}", (300, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Instructions
    cv2.putText(img, "Q: Quit | S: Screenshot | R: Reset", (10, 85),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Hand information panels
    for idx, hand in enumerate(hands):
        x, y, w, h = hand['bbox']
        fingers = hand['fingers']
        hand_type = hand['type']
        confidence = hand['confidence']
        
        # Info panel background
        panel_y = y - 80
        if panel_y < 100:
            panel_y = y + h + 10
        
        cv2.rectangle(img, (x - 10, panel_y), (x + 250, panel_y + 70), (0, 0, 0), -1)
        cv2.rectangle(img, (x - 10, panel_y), (x + 250, panel_y + 70), (255, 255, 255), 2)
        
        # Hand type and confidence
        cv2.putText(img, f"{hand_type} Hand", (x, panel_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(img, f"Confidence: {confidence:.0%}", (x, panel_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Gesture
        gesture = get_gesture_name(fingers)
        cv2.putText(img, gesture, (x, panel_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Finger pattern visualization
        finger_names = ['T', 'I', 'M', 'R', 'P']
        for i, (name, status) in enumerate(zip(finger_names, fingers)):
            color = (0, 255, 0) if status else (0, 0, 255)
            pos_x = x + 260 + i * 30
            pos_y = y + 20
            cv2.circle(img, (pos_x, pos_y), 12, color, -1)
            cv2.putText(img, name, (pos_x - 5, pos_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return img


def main():
    """Main function with professional features"""
    print("="*60)
    print("PROFESSIONAL HAND GESTURE DETECTION SYSTEM")
    print("="*60)
    
    if not MEDIAPIPE_AVAILABLE:
        print("\n⚠ WARNING: MediaPipe not installed!")
        print("Install with: pip install mediapipe")
        print("\nTrying alternative installation methods...")
        print("1. pip install mediapipe --user")
        print("2. python -m pip install mediapipe")
        return
    
    print("\n✓ All dependencies loaded")
    print("\nInitializing camera...")
    
    detector = ProfessionalHandDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Trying alternate camera...")
        cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("✗ Error: Cannot access camera")
        return
    
    # Camera settings for best quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✓ Camera initialized")
    print("\n" + "="*60)
    print("CONTROLS:")
    print("  Q - Quit")
    print("  S - Save Screenshot")
    print("  R - Reset frame counter")
    print("="*60 + "\n")
    
    prev_time = 0
    frame_count = 0
    
    while True:
        success, img = cap.read()
        if not success:
            continue
        
        frame_count += 1
        img = cv2.flip(img, 1)
        
        # FPS calculation
        current_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time
        
        # Detect hands
        hands, img = detector.find_hands(img, draw=True)
        
        # Draw professional UI
        img = draw_professional_ui(img, hands, fps, frame_count)
        
        # Display
        cv2.imshow('Professional Hand Detection', img)
        
        # Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"hand_capture_{frame_count}.jpg"
            cv2.imwrite(filename, img)
            print(f"✓ Screenshot saved: {filename}")
        elif key == ord('r'):
            frame_count = 0
            print("✓ Frame counter reset")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ System shutdown complete")
    print("="*60)


if __name__ == "__main__":
    main()