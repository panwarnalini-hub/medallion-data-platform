"""
Reads Bronze data and applies cleaning/normalization transformations

Run with: python silver_transform.py

What this does:
1. Reads raw frames from bronze_data/
2. Centers all coordinates on wrist (wrist becomes 0,0,0)
3. Normalizes scale (hand size becomes consistent)
4. Smooths jittery points (rolling average)
5. Interpolates missing frames (fills gaps)
6. Resamples to consistent 15 FPS
7. Saves cleaned data to silver_data/
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

import numpy as np
from pydantic import BaseModel


# SCHEMAS

class SilverFrame(BaseModel):
    """Cleaned and normalized frame data."""
    frame_id: str
    session_id: str
    source_frame_id: str
    sequence_idx: int
    timestamp_aligned: float
    hand_detected: bool
    handedness: Optional[str] = None
    landmarks_normalized: Optional[list[dict]] = None
    quality_score: float
    source_quality_score: Optional[float] = None
    interpolated: bool
    processing_version: str = "1.0.0"


class SilverSessionInfo(BaseModel):
    """Silver session metadata."""
    session_id: str
    source_session_id: str
    processed_at: str
    bronze_frames: int
    silver_frames: int
    detection_rate: float
    interpolated_frames: int
    target_fps: int


# TRANSFORMATION FUNCTIONS

LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]


def landmarks_to_array(landmarks: list[dict]) -> np.ndarray:
    """Convert landmark list to numpy array (21, 3)."""
    arr = np.zeros((21, 3))
    for lm in landmarks:
        idx = lm["idx"]
        arr[idx] = [lm["x"], lm["y"], lm["z"]]
    return arr


def array_to_landmarks(arr: np.ndarray) -> list[dict]:
    """Convert numpy array back to landmark list."""
    landmarks = []
    for i in range(21):
        landmarks.append({
            "idx": i,
            "name": LANDMARK_NAMES[i],
            "x": round(float(arr[i, 0]), 6),
            "y": round(float(arr[i, 1]), 6),
            "z": round(float(arr[i, 2]), 6)
        })
    return landmarks


def center_on_wrist(landmarks: np.ndarray) -> np.ndarray:
    """
    Step 1: Wrist Centering
    
    Subtract wrist position from all points.
    After this, wrist is at (0, 0, 0).
    
    Why: Same gesture at different screen positions now has same coordinates.
    """
    wrist = landmarks[0].copy()
    centered = landmarks - wrist
    return centered


def normalize_scale(landmarks: np.ndarray) -> np.ndarray:
    """
    Step 2: L2 Normalization (Scale)
    
    Scale all points so hand size is consistent.
    Uses wrist to middle finger MCP as reference length.
    
    Why: Hand close or far from camera now has same coordinates.
    """
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    
    ref_length = np.linalg.norm(middle_mcp - wrist)
    
    if ref_length < 0.001:
        return landmarks
    
    normalized = landmarks / ref_length
    return normalized


def compute_quality_score(landmarks: np.ndarray, handedness_confidence: Optional[float]) -> float:
    """Compute quality score based on detection confidence and landmark spread."""
    if handedness_confidence is None:
        return 0.0
    
    score = handedness_confidence
    
    # Penalize if landmarks too tight (bad detection)
    spread = np.std(landmarks[:, :2])
    if spread < 0.01:
        score *= 0.5
    
    # Penalize wild z values
    z_std = np.std(landmarks[:, 2])
    if z_std > 0.5:
        score *= 0.8
    
    return round(min(1.0, max(0.0, score)), 4)


def smooth_landmarks(frames: list[np.ndarray], window_size: int = 5) -> list[np.ndarray]:
    """
    Step 5: Temporal Smoothing
    
    Apply rolling average to reduce jitter.
    
    Why: Points jump around even when hand is still. Smoothing fixes this.
    """
    if len(frames) < window_size:
        return frames
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(frames)):
        start = max(0, i - half_window)
        end = min(len(frames), i + half_window + 1)
        
        window_frames = frames[start:end]
        avg = np.mean(window_frames, axis=0)
        smoothed.append(avg)
    
    return smoothed


def interpolate_missing(frames: list[dict], max_gap: int = 5) -> list[dict]:
    """
    Step 4: Missing Frame Interpolation
    
    Fill gaps where detection failed briefly.
    
    Why: Short detection failures shouldn't break the sequence.
    """
    result = []
    detected_indices = [i for i, f in enumerate(frames) if f["hand_detected"] and f.get("landmarks_normalized")]
    
    for i, frame in enumerate(frames):
        new_frame = frame.copy()
        
        if not frame["hand_detected"]:
            # Find prev and next detected
            prev_idx = None
            next_idx = None
            
            for di in detected_indices:
                if di < i:
                    prev_idx = di
                elif di > i and next_idx is None:
                    next_idx = di
                    break
            
            # Interpolate if gap is small
            if prev_idx is not None and next_idx is not None:
                gap = next_idx - prev_idx
                if gap <= max_gap:
                    t = (i - prev_idx) / gap
                    
                    prev_lm = frames[prev_idx]["landmarks_normalized"]
                    next_lm = frames[next_idx]["landmarks_normalized"]
                    
                    if prev_lm and next_lm:
                        prev_arr = landmarks_to_array(prev_lm)
                        next_arr = landmarks_to_array(next_lm)
                        
                        interp_arr = prev_arr + t * (next_arr - prev_arr)
                        
                        new_frame["hand_detected"] = True
                        new_frame["landmarks_normalized"] = array_to_landmarks(interp_arr)
                        new_frame["interpolated"] = True
                        new_frame["quality_score"] = 0.5
                        new_frame["source_quality_score"] = None
                        new_frame["handedness"] = frames[prev_idx].get("handedness")
        
        result.append(new_frame)
    
    return result


def resample_to_fps(frames: list[dict], target_fps: int = 15) -> list[dict]:
    """
    Step 6: Resample to Fixed FPS
    
    Convert variable frame rate to consistent 15 FPS.
    
    Why: Consistent timing makes downstream processing easier.
    """
    if not frames:
        return []
    
    # Parse timestamps
    for f in frames:
        ts = f["timestamp"]
        if isinstance(ts, str):
            f["_ts"] = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    
    start_time = frames[0]["_ts"]
    end_time = frames[-1]["_ts"]
    duration = (end_time - start_time).total_seconds()
    
    if duration <= 0:
        return frames[:1]
    
    target_interval = 1.0 / target_fps
    num_output = int(math.floor(duration * target_fps))
    
    resampled = []
    
    for i in range(num_output):
        target_time = i * target_interval
        
        # Find closest frame
        best_frame = None
        best_diff = float("inf")
        
        for f in frames:
            frame_time = (f["_ts"] - start_time).total_seconds()
            diff = abs(frame_time - target_time)
            if diff < best_diff:
                best_diff = diff
                best_frame = f
        
        if best_frame:
            new_frame = {k: v for k, v in best_frame.items() if k != "_ts"}
            new_frame["sequence_idx"] = i
            new_frame["timestamp_aligned"] = round(target_time, 4)
            resampled.append(new_frame)
    
    return resampled


# MAIN PROCESSING

def process_session(bronze_path: Path, silver_base: Path, target_fps: int = 15):
    """Process a single Bronze session into Silver."""
    
    print(f"\nProcessing: {bronze_path.name}")
    
    frames_file = bronze_path / "frames.jsonl"
    if not frames_file.exists():
        print("  No frames.jsonl found, skipping")
        return
    
    # Read bronze frames
    bronze_frames = []
    with open(frames_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                bronze_frames.append(json.loads(line))
    
    print(f"  Bronze frames: {len(bronze_frames)}")
    
    if not bronze_frames:
        return
    
    session_id = bronze_frames[0]["session_id"]
    
    # Step 1 & 2: Center and normalize
    print("  Centering on wrist + normalizing scale...")
    processed = []
    
    for bf in bronze_frames:
        frame_data = {
            "frame_id": str(uuid4()),
            "session_id": f"silver_{session_id}",
            "source_frame_id": bf["frame_id"],
            "timestamp": bf["timestamp"],
            "hand_detected": bf["hand_detected"],
            "handedness": bf.get("handedness"),
            "interpolated": False,
            "quality_score": 0.0,
            "source_quality_score": None,
            "landmarks_normalized": None
        }
        
        if bf["hand_detected"] and bf.get("landmarks"):
            arr = landmarks_to_array(bf["landmarks"])
            centered = center_on_wrist(arr)
            normalized = normalize_scale(centered)
            quality = compute_quality_score(normalized, bf.get("handedness_confidence"))
            
            frame_data["landmarks_normalized"] = array_to_landmarks(normalized)
            frame_data["quality_score"] = quality
            frame_data["source_quality_score"] = quality
        
        processed.append(frame_data)
    
    # Step 4: Interpolate missing
    print("  Interpolating missing frames...")
    interpolated = interpolate_missing(processed)
    interpolated_count = sum(1 for f in interpolated if f.get("interpolated", False))
    
    # Step 5: Smooth landmarks
    print("  Smoothing jitter...")
    detected = [
        f for f in interpolated
        if f["hand_detected"] and f.get("landmarks_normalized")
    ]
    if detected:
        arrays = [landmarks_to_array(f["landmarks_normalized"]) for f in interpolated if f["hand_detected"]]
        smoothed = smooth_landmarks(arrays, window_size=5)
        
        smooth_idx = 0
        for f in interpolated:
            if f["hand_detected"] and f.get("landmarks_normalized"):
                f["landmarks_normalized"] = array_to_landmarks(smoothed[smooth_idx])
                smooth_idx += 1

    
    # Step 6: Resample
    print(f"  Resampling to {target_fps} FPS...")
    resampled = resample_to_fps(interpolated, target_fps=target_fps)
    
    # Create Silver frames
    silver_frames = []
    for i, f in enumerate(resampled):
        sf = SilverFrame(
            frame_id=f.get("frame_id", str(uuid4())),
            session_id=f.get("session_id", f"silver_{session_id}"),
            source_frame_id=f.get("source_frame_id", ""),
            sequence_idx=f.get("sequence_idx", i),
            timestamp_aligned=f.get("timestamp_aligned", i / target_fps),
            hand_detected=f["hand_detected"],
            handedness=f.get("handedness"),
            landmarks_normalized=f.get("landmarks_normalized"),
            quality_score=f.get("quality_score", 0.0),
            source_quality_score=f.get("source_quality_score"),
            interpolated=f.get("interpolated", False)
        )
        silver_frames.append(sf)
    
    # Save
    silver_folder = silver_base / f"silver_{session_id}"
    silver_folder.mkdir(parents=True, exist_ok=True)
    
    with open(silver_folder / "frames.jsonl", "w", encoding="utf-8") as f:
        for sf in silver_frames:
            f.write(sf.model_dump_json() + "\n")
    
    detected_count = sum(1 for f in silver_frames if f.hand_detected)
    rate = round(detected_count / len(silver_frames) * 100, 2) if silver_frames else 0
    
    session_info = SilverSessionInfo(
        session_id=f"silver_{session_id}",
        source_session_id=session_id,
        processed_at=datetime.now(timezone.utc).isoformat(),
        bronze_frames=len(bronze_frames),
        silver_frames=len(silver_frames),
        detection_rate=rate,
        interpolated_frames=interpolated_count,
        target_fps=target_fps
    )
    
    with open(silver_folder / "session_info.json", "w", encoding="utf-8") as f:
        f.write(session_info.model_dump_json(indent=2))
    
    print(f"\n  Output: {silver_folder}")
    print(f"  Frames: {len(bronze_frames)} -> {len(silver_frames)}")
    print(f"  Detection rate: {rate}%")
    print(f"  Interpolated: {interpolated_count}")


def main():
    print("GESTURE PIPELINE : SILVER TRANSFORM")
    
    bronze_base = Path("bronze_data")
    silver_base = Path("silver_data")
    
    if not bronze_base.exists():
        print("\nNo bronze_data folder found!")
        print("Run bronze_ingestion.py first.")
        return
    
    sessions = [d for d in bronze_base.iterdir() if d.is_dir() and d.name.startswith("session_")]
    
    if not sessions:
        print("\nNo sessions found in bronze_data/")
        return
    
    print(f"\nFound {len(sessions)} session(s)")
    
    for session_path in sessions:
        process_session(session_path, silver_base)
    
    print("DONE")
    print("\nCheck silver_data/ folder.")
    print("\nTransformations applied:")
    print("  - Wrist centered (wrist = 0,0,0)")
    print("  - Scale normalized")
    print("  - Jitter smoothed")
    print("  - Missing frames interpolated")
    print("  - Resampled to 15 FPS")


if __name__ == "__main__":
    main()
