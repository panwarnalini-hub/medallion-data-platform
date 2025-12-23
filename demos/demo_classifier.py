"""
Run with: python demo_classifier.py
Press 'q' to quit
"""

import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from gesture_classifier import Gesture, GestureClassifier


def landmarks_to_array(hand_landmarks) -> np.ndarray:
    arr = np.zeros((21, 3))
    for i, lm in enumerate(hand_landmarks):
        arr[i] = [lm.x, lm.y, lm.z]
    return arr


def download_model():
    model_path = Path("hand_landmarker.task")
    if not model_path.exists():
        print("Downloading model...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
    return str(model_path)


class GestureDemo:
    COLORS = {
        Gesture.NONE: (128, 128, 128),
        Gesture.OPEN_PALM: (0, 255, 0),
        Gesture.FIST: (0, 0, 255),
        Gesture.POINTING: (255, 255, 0),
        Gesture.THUMBS_UP: (0, 255, 255),
        Gesture.PEACE: (255, 0, 255),
        Gesture.SWIPE_LEFT: (255, 128, 0),
        Gesture.SWIPE_RIGHT: (0, 128, 255),
    }

    def __init__(self):
        self.classifier = GestureClassifier()
        self.cap = None
        self.detector = None
        self.latest_result = None
        self.counts = {g: 0 for g in Gesture}
        self.frames = 0

    def _callback(self, result, output_image, timestamp_ms):
        self.latest_result = result

    def start(self) -> bool:
        print("\nSetting up...")
        model_path = download_model()

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            result_callback=self._callback
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("ERROR: Could not open webcam")
            return False

        print("Ready!")
        return True

    def process_frame(self, frame, ts: int):
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.detector.detect_async(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts
        )

        result = self.latest_result
        gesture = Gesture.NONE
        bends = {}

        if result and result.hand_landmarks:
            lm = result.hand_landmarks[0]
            arr = landmarks_to_array(lm)
            gesture, bends = self.classifier.classify(arr)

            h, w = frame.shape[:2]
            color = self.COLORS[gesture]

            for p in lm:
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 5, color, -1)

            conns = [
                (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),
                (15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
            ]

            for s, e in conns:
                cv2.line(
                    frame,
                    (int(lm[s].x*w), int(lm[s].y*h)),
                    (int(lm[e].x*w), int(lm[e].y*h)),
                    color,
                    2
                )

        if gesture != Gesture.NONE:
            self.counts[gesture] += 1
        self.frames += 1

        return frame, gesture, bends

    def draw_ui(self, frame, gesture, bends):
        h, w = frame.shape[:2]
        color = self.COLORS[gesture]

        # Gesture display box
        cv2.rectangle(frame, (10, 10), (280, 90), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (280, 90), color, 2)
        cv2.putText(
            frame,
            gesture.value.replace("_", " "),
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2
        )

        # Finger bend bars
        y = 110
        for finger in ["index", "middle", "ring", "pinky"]:
            if finger in bends:
                b = bends[finger]
                cv2.rectangle(frame, (10, y), (110, y+12), (50,50,50), -1)
                cv2.rectangle(
                    frame,
                    (10, y),
                    (10 + int(b * 100), y + 12),
                    (0,0,255) if b > 0.4 else (0,255,0),
                    -1
                )
                cv2.putText(
                    frame,
                    f"{finger}: {'BENT' if b > 0.4 else 'OPEN'}",
                    (120, y+10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (200,200,200),
                    1
                )
                y += 18

        # Instructions at bottom
        cv2.putText(
            frame,
            "GESTURES: Swipe L/R | Thumbs Up | Fist | Open Palm | Point | Peace",
            (10, h - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1
        )
        cv2.putText(
            frame,
            "Press 'q' to QUIT",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )

        return frame

    def run(self):
        print("GESTURE CLASSIFIER")
        print("GESTURES:")
        print("  👈 Swipe LEFT  - move hand left quickly")
        print("  👉 Swipe RIGHT - move hand right quickly")
        print("  👍 Thumbs UP   - thumb up, fingers closed")
        print("  ✊ Fist        - all fingers closed")
        print("  ✋ Open Palm   - all fingers open")
        print("  👆 Pointing    - only index finger up")
        print("  ✌️  Peace       - index + middle up")
        print("Press 'q' to QUIT")
        
        ts = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            ts += 33
            frame, gesture, bends = self.process_frame(frame, ts)
            frame = self.draw_ui(frame, gesture, bends)

            cv2.imshow("Gesture Classifier", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def stop(self):
        if self.detector:
            self.detector.close()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Print stats
        print("\n" + "=" * 40)
        print("SESSION STATS")
        print("=" * 40)
        print(f"Total frames: {self.frames}")
        print("\nGestures detected:")
        for g, count in self.counts.items():
            if count > 0:
                print(f"  {g.value}: {count}")


def main():
    demo = GestureDemo()
    try:
        if demo.start():
            demo.run()
    finally:
        demo.stop()


if __name__ == "__main__":
    main()
