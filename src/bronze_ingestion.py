"""
Captures raw hand landmarks from webcam using MediaPipe Tasks API

Run with: python demo_bronze.py
"""

import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from pydantic import BaseModel


# Landmark names for the 21 hand points
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]


class BronzeFrame(BaseModel):
    """Raw frame data : no transformations"""
    frame_id: str
    session_id: str
    timestamp: str
    frame_number: int
    hand_detected: bool
    handedness: Optional[str] = None
    landmarks: Optional[list[dict]] = None
    handedness_confidence: Optional[float] = None  # Confidence in handedness classification
    min_hand_detection_confidence: float = 0.7  # Threshold used for detection
    min_tracking_confidence: float = 0.5  # Threshold used for tracking
    drop_reason: Optional[str] = None  # Why no hand detected (for debugging)
    processing_time_ms: int


class SessionInfo(BaseModel):
    """Capture session metadata."""
    session_id: str
    started_at: str
    ended_at: Optional[str] = None
    total_frames: int = 0
    frames_with_hand: int = 0


class BronzeStorage:
    """Handles writing frames to JSONL files."""
    
    def __init__(self, base_folder: str = "bronze_data"):
        self.base_folder = Path(base_folder)
        self.session_folder = None
        self.frames_file = None
        self.session_info = None
    
    def start_session(self) -> str:
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_folder = self.base_folder / session_id
        self.session_folder.mkdir(parents=True, exist_ok=True)
        
        self.session_info = SessionInfo(
            session_id=session_id,
            started_at=datetime.now(timezone.utc).isoformat()
        )
        
        frames_path = self.session_folder / "frames.jsonl"
        self.frames_file = open(frames_path, "w", encoding="utf-8")
        
        print(f"\nSession folder: {self.session_folder}")
        return session_id
    
    def save_frame(self, frame: BronzeFrame):
        self.frames_file.write(frame.model_dump_json() + "\n")
        self.session_info.total_frames += 1
        if frame.hand_detected:
            self.session_info.frames_with_hand += 1
    
    def end_session(self):
        if self.frames_file:
            self.frames_file.close()
        
        if self.session_info and self.session_folder:
            self.session_info.ended_at = datetime.now(timezone.utc).isoformat()
            info_path = self.session_folder / "session_info.json"
            with open(info_path, "w", encoding="utf-8") as f:
                f.write(self.session_info.model_dump_json(indent=2))
            
            total = self.session_info.total_frames
            detected = self.session_info.frames_with_hand
            rate = (detected / total * 100) if total > 0 else 0
            
            print(f"\nSession complete:")
            print(f"  Total frames: {total}")
            print(f"  Hand detected: {detected} ({rate:.1f}%)")
            print(f"  Data saved to: {self.session_folder}")


def download_model():
    """Download the hand landmarker model if not present."""
    model_path = Path("hand_landmarker.task")
    if not model_path.exists():
        print("Downloading hand landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded.")
    return str(model_path)


class GestureCapture:
    """Webcam capture with MediaPipe hand detection using Tasks API."""
    
    def __init__(self):
        self.storage = BronzeStorage()
        self.session_id = None
        self.frame_number = 0
        self.cap = None
        self.detector = None
        self.latest_result = None
    
    def _result_callback(self, result, output_image, timestamp_ms):
        """Callback for async hand detection results."""
        self.latest_result = result
    
    def start(self) -> bool:
        print("\nSetting up MediaPipe...")
        
        # Download model if needed
        model_path = download_model()
        
        # Create hand landmarker
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            result_callback=self._result_callback
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        print("Opening webcam...")
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("ERROR: Could not open webcam")
            return False
        
        print("Webcam ready")
        self.session_id = self.storage.start_session()
        self.frame_number = 0
        return True
    
    def process_frame(self, frame, timestamp_ms: int):
        start_time = time.perf_counter()
        
        # Convert to RGB and create MediaPipe Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Run detection asynchronously
        self.detector.detect_async(mp_image, timestamp_ms)
        
        # Use latest result (may be from previous frame due to async)
        result = self.latest_result
        
        # Determine detection status and drop reason
        hand_detected = False
        drop_reason = None
        landmarks = None
        handedness = None
        confidence = None
        
        if result is None:
            drop_reason = "no_result_yet"
        elif len(result.hand_landmarks) == 0:
            drop_reason = "filtered_by_mediapipe_thresholds"
        else:
            hand_detected = True
            hand_landmarks = result.hand_landmarks[0]
            hand_info = result.handedness[0][0]
            
            landmarks = []
            for i, lm in enumerate(hand_landmarks):
                landmarks.append({
                    "idx": i,
                    "name": LANDMARK_NAMES[i],
                    "x": round(lm.x, 6),
                    "y": round(lm.y, 6),
                    "z": round(lm.z, 6)
                })
            
            handedness = hand_info.category_name.upper()
            confidence = round(hand_info.score, 4)
            
            # Draw landmarks on frame
            h, w = frame.shape[:2]
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            
            # Draw connections
            connections = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),
                (0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20),
                (5,9),(9,13),(13,17)
            ]
            for start, end in connections:
                x1, y1 = int(hand_landmarks[start].x * w), int(hand_landmarks[start].y * h)
                x2, y2 = int(hand_landmarks[end].x * w), int(hand_landmarks[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        processing_ms = int((time.perf_counter() - start_time) * 1000)
        
        bronze_frame = BronzeFrame(
            frame_id=str(uuid4()),
            session_id=self.session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            frame_number=self.frame_number,
            hand_detected=hand_detected,
            handedness=handedness,
            landmarks=landmarks,
            handedness_confidence=confidence,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            drop_reason=drop_reason,
            processing_time_ms=processing_ms
        )
        
        self.storage.save_frame(bronze_frame)
        self.frame_number += 1
        
        return frame, bronze_frame
    
    def run(self, duration_seconds: int = 30):
        print(f"\nCapturing for {duration_seconds} seconds...")
        print("Show your hand to the camera. Press 'q' to stop early.\n")
        
        start_time = time.time()
        frame_timestamp = 0
        
        while True:
            if (time.time() - start_time) >= duration_seconds:
                print(f"\nTime limit reached ({duration_seconds}s)")
                break
            
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read from webcam")
                break
            
            frame_timestamp += 33  # ~30fps
            annotated, bronze_frame = self.process_frame(frame, frame_timestamp)
            
            # Display status
            if bronze_frame.hand_detected:
                status = f"{bronze_frame.handedness} hand detected"
                color = (0, 255, 0)
            else:
                status = "No hand"
                color = (0, 0, 255)
            
            cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(annotated, f"Frame {bronze_frame.frame_number}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(annotated, "Press 'q' to stop", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Gesture Capture", annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopped by user")
                break
    
    def stop(self):
        if self.detector:
            self.detector.close()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.storage.end_session()


def main():
    print("GESTURE PIPELINE : BRONZE INGESTION")
    
    capture = GestureCapture()
    
    try:
        if capture.start():
            capture.run(duration_seconds=30)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        capture.stop()
    
    print("\nDone. Check bronze_data folder for your captured data.")


if __name__ == "__main__":
    main()
