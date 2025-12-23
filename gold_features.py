"""
Computes ML-ready features from Silver data

Run with: python gold_features.py

What this does:
1. Reads normalized frames from silver_data/
2. Computes geometric features (finger angles, spreads)
3. Computes motion features (velocity, acceleration)
4. Computes temporal aggregates (rolling stats)
5. Outputs feature vectors ready for ML models
6. Saves to gold_data/
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

class GoldFeatureVector(BaseModel):
    """ML-ready feature vector for a time window."""
    feature_id: str
    session_id: str
    window_start_idx: int
    window_end_idx: int
    timestamp_start: float
    timestamp_end: float
    
    # Feature vector (all features concatenated)
    features: list[float]
    
    # Feature manifest (which indices map to which features)
    feature_names: list[str]
    
    # Metadata
    frames_in_window: int
    detection_rate: float
    avg_quality_score: float
    
    schema_version: str = "1.0.0"


class GoldSessionInfo(BaseModel):
    """Gold session metadata."""
    session_id: str
    source_session_id: str
    processed_at: str
    silver_frames: int
    feature_vectors: int
    window_size: int
    window_stride: int
    feature_count: int
    feature_names: list[str]

# LANDMARK INDICES (for readable code)

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


# FEATURE COMPUTATION FUNCTIONS

def landmarks_to_array(landmarks: list[dict]) -> np.ndarray:
    """Convert landmark list to numpy array (21, 3)."""
    arr = np.zeros((21, 3))
    for lm in landmarks:
        arr[lm["idx"]] = [lm["x"], lm["y"], lm["z"]]
    return arr


def compute_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Compute angle at p2 formed by p1-p2-p3.
    Returns angle in radians [0, pi].
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    return math.acos(cos_angle)


def compute_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(p1 - p2))


# GEOMETRIC FEATURES (from single frame)

def compute_finger_angles(landmarks: np.ndarray) -> dict:
    """
    Compute bend angle for each finger.
    
    For each finger, we measure the angle at the PIP joint
    (or MCP for thumb). Straight finger = ~pi, bent = smaller.
    
    Returns dict with 5 angles (one per finger).
    """
    angles = {}
    
    # Thumb: angle at IP joint
    angles["thumb_bend"] = compute_angle(
        landmarks[THUMB_CMC], landmarks[THUMB_MCP], landmarks[THUMB_IP]
    )
    
    # Index: angle at PIP joint
    angles["index_bend"] = compute_angle(
        landmarks[INDEX_MCP], landmarks[INDEX_PIP], landmarks[INDEX_DIP]
    )
    
    # Middle: angle at PIP joint
    angles["middle_bend"] = compute_angle(
        landmarks[MIDDLE_MCP], landmarks[MIDDLE_PIP], landmarks[MIDDLE_DIP]
    )
    
    # Ring: angle at PIP joint
    angles["ring_bend"] = compute_angle(
        landmarks[RING_MCP], landmarks[RING_PIP], landmarks[RING_DIP]
    )
    
    # Pinky: angle at PIP joint
    angles["pinky_bend"] = compute_angle(
        landmarks[PINKY_MCP], landmarks[PINKY_PIP], landmarks[PINKY_DIP]
    )
    
    return angles


def compute_finger_spreads(landmarks: np.ndarray) -> dict:
    """
    Compute spread between adjacent fingertips.
    
    Higher value = fingers spread apart.
    Lower value = fingers together.
    """
    spreads = {}
    
    spreads["thumb_index_spread"] = compute_distance(
        landmarks[THUMB_TIP], landmarks[INDEX_TIP]
    )
    spreads["index_middle_spread"] = compute_distance(
        landmarks[INDEX_TIP], landmarks[MIDDLE_TIP]
    )
    spreads["middle_ring_spread"] = compute_distance(
        landmarks[MIDDLE_TIP], landmarks[RING_TIP]
    )
    spreads["ring_pinky_spread"] = compute_distance(
        landmarks[RING_TIP], landmarks[PINKY_TIP]
    )
    
    return spreads


def compute_palm_metrics(landmarks: np.ndarray) -> dict:
    """
    Compute palm-related measurements.
    """
    metrics = {}
    
    # Palm width: distance between index MCP and pinky MCP
    metrics["palm_width"] = compute_distance(
        landmarks[INDEX_MCP], landmarks[PINKY_MCP]
    )
    
    # Palm length: distance from wrist to middle MCP
    metrics["palm_length"] = compute_distance(
        landmarks[WRIST], landmarks[MIDDLE_MCP]
    )
    
    # Hand openness: average distance from wrist to all fingertips
    tip_distances = [
        compute_distance(landmarks[WRIST], landmarks[tip])
        for tip in FINGERTIPS
    ]
    metrics["hand_openness"] = float(np.mean(tip_distances))
    
    return metrics


def compute_fingertip_heights(landmarks: np.ndarray) -> dict:
    """
    Compute relative height (y-coordinate) of each fingertip.
    
    Useful for detecting pointing up vs down.
    """
    heights = {}
    
    wrist_y = landmarks[WRIST][1]
    
    heights["thumb_height"] = landmarks[THUMB_TIP][1] - wrist_y
    heights["index_height"] = landmarks[INDEX_TIP][1] - wrist_y
    heights["middle_height"] = landmarks[MIDDLE_TIP][1] - wrist_y
    heights["ring_height"] = landmarks[RING_TIP][1] - wrist_y
    heights["pinky_height"] = landmarks[PINKY_TIP][1] - wrist_y
    
    return heights


def compute_frame_features(landmarks: np.ndarray) -> dict:
    """
    Compute all geometric features for a single frame.
    """
    features = {}
    
    features.update(compute_finger_angles(landmarks))
    features.update(compute_finger_spreads(landmarks))
    features.update(compute_palm_metrics(landmarks))
    features.update(compute_fingertip_heights(landmarks))
    
    return features


# MOTION FEATURES (from frame sequence)

def compute_velocities(landmarks_sequence: list[np.ndarray], dt: float) -> dict:
    """
    Compute velocity features from a sequence of frames.
    
    dt: time between frames in seconds (1/fps)
    
    Returns average velocity magnitude for key points.
    """
    if len(landmarks_sequence) < 2:
        return {
            "wrist_velocity": 0.0,
            "index_tip_velocity": 0.0,
            "thumb_tip_velocity": 0.0,
            "avg_fingertip_velocity": 0.0
        }
    # Palm scale for velocity normalization
    palm_scale = (
        np.linalg.norm(
        landmarks_sequence[0][INDEX_MCP]
        - landmarks_sequence[0][PINKY_MCP]
        ) + 1e-6
    )
    velocities = {
        "wrist": [],
        "index_tip": [],
        "thumb_tip": [],
        "fingertips": []
    }
    
    for i in range(1, len(landmarks_sequence)):
        prev = landmarks_sequence[i - 1]
        curr = landmarks_sequence[i]
        
        # Wrist velocity
        wrist_vel = (np.linalg.norm(curr[WRIST] - prev[WRIST]) / dt) / palm_scale
        velocities["wrist"].append(wrist_vel)
        
        # Index tip velocity
        index_vel = (np.linalg.norm(curr[INDEX_TIP] - prev[INDEX_TIP]) / dt) / palm_scale
        velocities["index_tip"].append(index_vel)
        
        # Thumb tip velocity
        thumb_vel = (np.linalg.norm(curr[THUMB_TIP] - prev[THUMB_TIP]) / dt) / palm_scale
        velocities["thumb_tip"].append(thumb_vel)
        
        # Average fingertip velocity
        tip_vels = [
            (np.linalg.norm(curr[tip] - prev[tip]) / dt) / palm_scale
            for tip in FINGERTIPS
        ]
        velocities["fingertips"].append(np.mean(tip_vels))
    
    return {
        "wrist_velocity": float(np.mean(velocities["wrist"])),
        "index_tip_velocity": float(np.mean(velocities["index_tip"])),
        "thumb_tip_velocity": float(np.mean(velocities["thumb_tip"])),
        "avg_fingertip_velocity": float(np.mean(velocities["fingertips"]))
    }


def compute_acceleration(landmarks_sequence: list[np.ndarray], dt: float) -> dict:
    """
    Compute acceleration (change in velocity) for key points.
    """
    if len(landmarks_sequence) < 3:
        return {
            "wrist_acceleration": 0.0,
            "avg_fingertip_acceleration": 0.0
        }
    
    # Compute velocities first
    wrist_vels = []
    tip_vels = []
    
    for i in range(1, len(landmarks_sequence)):
        prev = landmarks_sequence[i - 1]
        curr = landmarks_sequence[i]
        
        wrist_vels.append(np.linalg.norm(curr[WRIST] - prev[WRIST]) / dt)
        
        tip_vel = np.mean([
            np.linalg.norm(curr[tip] - prev[tip]) / dt
            for tip in FINGERTIPS
        ])
        tip_vels.append(tip_vel)
    
    # Compute acceleration (change in velocity)
    wrist_accels = [
        abs(wrist_vels[i] - wrist_vels[i-1]) / dt
        for i in range(1, len(wrist_vels))
    ]
    
    tip_accels = [
        abs(tip_vels[i] - tip_vels[i-1]) / dt
        for i in range(1, len(tip_vels))
    ]
    
    return {
        "wrist_acceleration": float(np.mean(wrist_accels)) if wrist_accels else 0.0,
        "avg_fingertip_acceleration": float(np.mean(tip_accels)) if tip_accels else 0.0
    }


# TEMPORAL AGGREGATES (statistics over window)

def compute_temporal_aggregates(frame_features: list[dict]) -> dict:
    """
    Compute statistical aggregates over a window of frames.
    
    For each geometric feature, compute: mean, std, min, max
    """
    if not frame_features:
        return {}
    
    aggregates = {}
    
    # Get all feature names from first frame
    feature_names = list(frame_features[0].keys())
    
    for name in feature_names:
        values = [f[name] for f in frame_features if name in f]
        
        if values:
            aggregates[f"{name}_mean"] = float(np.mean(values))
            aggregates[f"{name}_std"] = float(np.std(values))
            aggregates[f"{name}_min"] = float(np.min(values))
            aggregates[f"{name}_max"] = float(np.max(values))
    
    return aggregates


# MAIN FEATURE EXTRACTION

def extract_window_features(
    frames: list[dict],
    fps: int = 15
) -> tuple[list[float], list[str]]:
    """
    Extract all features from a window of frames.
    
    Returns:
        features: list of feature values
        names: list of feature names (same order)
    """
    # Get landmark arrays for detected frames
    detected_frames = [f for f in frames if f["hand_detected"] and f.get("landmarks_normalized")]
    
    if not detected_frames:
        # Return zeros if no detection
        dummy_features = {
            "thumb_bend": 0, "index_bend": 0, "middle_bend": 0, "ring_bend": 0, "pinky_bend": 0,
            "thumb_index_spread": 0, "index_middle_spread": 0, "middle_ring_spread": 0, "ring_pinky_spread": 0,
            "palm_width": 0, "palm_length": 0, "hand_openness": 0,
            "thumb_height": 0, "index_height": 0, "middle_height": 0, "ring_height": 0, "pinky_height": 0,
            "wrist_velocity": 0, "index_tip_velocity": 0, "thumb_tip_velocity": 0, "avg_fingertip_velocity": 0,
            "wrist_acceleration": 0, "avg_fingertip_acceleration": 0,
            "detection_ratio": 0
        }
        return list(dummy_features.values()), list(dummy_features.keys())
    
    landmarks_arrays = [
        landmarks_to_array(f["landmarks_normalized"])
        for f in detected_frames
    ]
    
    dt = 1.0 / fps
    
    # Compute geometric features for each frame
    frame_features = [compute_frame_features(lm) for lm in landmarks_arrays]
    
    # Compute motion features
    motion_features = compute_velocities(landmarks_arrays, dt)
    motion_features.update(compute_acceleration(landmarks_arrays, dt))
    
    # Compute temporal aggregates
    temporal_agg = compute_temporal_aggregates(frame_features)
    
    # Use last frame's geometric features as "current" state
    current_features = frame_features[-1] if frame_features else {}
    
    # Combine all features
    all_features = {}
    
    # Current geometric state
    for name, value in current_features.items():
        all_features[name] = value
    
    # Motion features
    for name, value in motion_features.items():
        all_features[name] = value
    
    # Temporal aggregates (just std to capture variability)
    for name in ["thumb_bend", "index_bend", "hand_openness"]:
        if f"{name}_std" in temporal_agg:
            all_features[f"{name}_variability"] = temporal_agg[f"{name}_std"]
    
    # Detection ratio in window
    all_features["detection_ratio"] = len(detected_frames) / len(frames)
    
    # Convert to lists
    names = sorted(all_features.keys())
    values = [round(float(all_features[n]), 6) for n in names]
    
    return values, names


def process_session(silver_path: Path, gold_base: Path, window_size: int = 15, window_stride: int = 5):
    """
    Process a Silver session into Gold feature vectors.
    
    window_size: Number of frames per window (15 frames = 1 second at 15fps)
    window_stride: Step between windows (5 = overlap of 10 frames)
    """
    print(f"\nProcessing: {silver_path.name}")
    
    frames_file = silver_path / "frames.jsonl"
    if not frames_file.exists():
        print("  No frames.jsonl found, skipping")
        return
    
    # Read silver frames
    silver_frames = []
    with open(frames_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                silver_frames.append(json.loads(line))
    
    print(f"  Silver frames: {len(silver_frames)}")
    
    if len(silver_frames) < window_size:
        print(f"  Not enough frames for window size {window_size}")
        return
    
    session_id = silver_frames[0]["session_id"]
    
    # Extract features for each window
    print(f"  Extracting features (window={window_size}, stride={window_stride})...")
    
    feature_vectors = []
    feature_names = None
    
    for start_idx in range(0, len(silver_frames) - window_size + 1, window_stride):
        end_idx = start_idx + window_size
        window_frames = silver_frames[start_idx:end_idx]
        
        # Extract features
        features, names = extract_window_features(window_frames, fps=15)
        
        if feature_names is None:
            feature_names = names
        
        # Compute window metadata
        detected_count = sum(1 for f in window_frames if f["hand_detected"])
        quality_scores = [f["quality_score"] for f in window_frames if f["hand_detected"]]
        avg_quality = float(np.mean(quality_scores)) if quality_scores else 0.0
        
        fv = GoldFeatureVector(
            feature_id=str(uuid4()),
            session_id=f"gold_{session_id}",
            window_start_idx=start_idx,
            window_end_idx=end_idx,
            timestamp_start=window_frames[0]["timestamp_aligned"],
            timestamp_end=window_frames[-1]["timestamp_aligned"],
            features=features,
            feature_names=names,
            frames_in_window=len(window_frames),
            detection_rate=round(detected_count / len(window_frames), 4),
            avg_quality_score=round(avg_quality, 4)
        )
        feature_vectors.append(fv)
    
    print(f"  Feature vectors: {len(feature_vectors)}")
    print(f"  Features per vector: {len(feature_names)}")
    
    # Save
    gold_folder = gold_base / f"gold_{session_id}"
    gold_folder.mkdir(parents=True, exist_ok=True)
    
    # Write feature vectors
    with open(gold_folder / "features.jsonl", "w", encoding="utf-8") as f:
        for fv in feature_vectors:
            f.write(fv.model_dump_json() + "\n")
    
    # Write session info
    session_info = GoldSessionInfo(
        session_id=f"gold_{session_id}",
        source_session_id=session_id,
        processed_at=datetime.now(timezone.utc).isoformat(),
        silver_frames=len(silver_frames),
        feature_vectors=len(feature_vectors),
        window_size=window_size,
        window_stride=window_stride,
        feature_count=len(feature_names),
        feature_names=feature_names
    )
    
    with open(gold_folder / "session_info.json", "w", encoding="utf-8") as f:
        f.write(session_info.model_dump_json(indent=2))
    
    # Write feature names for reference
    with open(gold_folder / "feature_manifest.json", "w", encoding="utf-8") as f:
        manifest = {
            "version": "1.0.0",
            "feature_count": len(feature_names),
            "features": {name: i for i, name in enumerate(feature_names)}
        }
        json.dump(manifest, f, indent=2)
    
    print(f"\n  Output: {gold_folder}")
    print(f"  Feature names: {feature_names[:5]}... (and {len(feature_names)-5} more)")


def main():
    print("GESTURE PIPELINE : GOLD FEATURES")
    
    silver_base = Path("silver_data")
    gold_base = Path("gold_data")
    
    if not silver_base.exists():
        print("\nNo silver_data folder found!")
        print("Run silver_transform.py first.")
        return
    
    sessions = [d for d in silver_base.iterdir() if d.is_dir() and d.name.startswith("silver_")]
    
    if not sessions:
        print("\nNo sessions found in silver_data/")
        return
    
    print(f"\nFound {len(sessions)} session(s)")
    
    for session_path in sessions:
        process_session(
            session_path, 
            gold_base,
            window_size=15,   # 1 second window at 15fps
            window_stride=5   # 0.33 second step
        )
    
    print("DONE")
    print("\nCheck gold_data/ folder.")
    print("\nFeatures computed:")
    print("  - Finger bend angles (5)")
    print("  - Finger spreads (4)")
    print("  - Palm metrics (3)")
    print("  - Fingertip heights (5)")
    print("  - Velocities (4)")
    print("  - Accelerations (2)")
    print("  - Variability measures (3)")
    print("  - Detection ratio (1)")
    print("\nReady for ML training!")


if __name__ == "__main__":
    main()
