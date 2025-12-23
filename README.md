# Medallion Data Platform

A production-grade data platform implementing the Medallion (Bronze-Silver-Gold) architecture for deterministic ingestion, transformation, and feature engineering of multimodal signals.

The system converts raw vision-based gesture data into analytics- and ML-ready feature vectors with explicit schemas, reproducible transformations, and clear data contracts.

---

## Architecture

This platform follows a strict Medallion architecture. Each layer is isolated and independently executable.


* **Raw Input:** (Webcam / MediaPipe)
* **Bronze:** Raw, append-only ingestion.
* **Silver:** Normalized, time-aligned, quality-scored data.
* **Gold:** Windowed, ML-ready feature vectors.

### Core Guarantees

* **Append-only raw data**
* **Deterministic transformations**
* **Explicit schemas at every layer**
* **Idempotent reprocessing**
* **Separation of ingestion, transformation, and feature logic**

---

## Repository Structure

    medallion-data-platform/
    ├── bronze_data/              # Raw sessions (append-only)
    ├── silver_data/              # Cleaned & normalized sessions
    ├── gold_data/                # ML-ready feature vectors
    │
    ├── bronze_ingestion.py       # Webcam - Bronze
    ├── silver_transform.py       # Bronze - Silver
    ├── gold_features.py          # Silver - Gold
    │
    ├── gesture_classifier.py     # Deterministic gesture logic
    ├── demo_classifier.py        # Real-time classification demo
    ├── demo_kiosk.py             # Gesture-driven UI demo
    │
    ├── run-gesture-pipeline.ps1
    ├── run-gesture-demo.ps1
    ├── requirements.txt
    └── README.md

---

## Pipeline Layers

### Bronze Layer : Ingestion
**Responsibility:** Capture raw data with zero transformation.

* MediaPipe Tasks API for hand landmark detection
* Append-only JSONL storage
* Detection confidence, handedness, and drop reasons preserved
* No mutation of raw signals

**Run:**

    python bronze_ingestion.py

### Silver Layer : Transformation
**Responsibility:** Normalize raw data into a consistent, analyzable form.

**Transformations:**
* Wrist centering (origin normalization)
* L2 scale normalization
* Temporal smoothing
* Missing-frame interpolation
* Fixed-rate resampling (15 FPS)
* Per-frame quality scoring

**Run:**

    python silver_transform.py

### Gold Layer : Feature Engineering
**Responsibility:** Produce ML-ready features with stable semantics.

**Features:**
* Finger bend angles
* Finger spreads
* Palm geometry
* Fingertip heights
* Velocity and acceleration
* Temporal variability
* Detection ratio

**Windowing:**
* Window size: 15 frames
* Stride: 5 frames

**Run:**

    python gold_features.py

---

## Demos

### Gesture Classifier
Real-time gesture classification with visual feedback.

    python demo_classifier.py

### Touchless Kiosk
End-to-end gesture-driven UI using WebSockets.

    python demo_kiosk.py

---

## Production Considerations

* Raw data is immutable and replayable
* Transformations are idempotent
* Feature definitions are versionable
* Clear separation between data capture, processing, and consumption
* Designed for batch or streaming execution models

## Intended Audience

* Data Engineers
* ML Platform Engineers
* Analytics Infrastructure Teams
* Engineers designing production data pipelines

## License

MIT License
