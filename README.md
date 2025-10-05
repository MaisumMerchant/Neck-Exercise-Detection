# Neck Exercise Detection

Tracks and counts neck exercises using head pose estimation. Supports both geometric and ML-based approaches.

## Installation

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn joblib matplotlib
```

## Files

**`neck_exercise_geo.py`** - Geometric method (no training needed)
- Uses OpenCV solvePnP for head pose estimation
- Run: `python neck_exercise_geo.py`

**`neck_exercise_ml_train.py`** - Train ML model
- Requires: `facemesh_landmarks_with_pose.csv`
- Outputs: `trained_model.pkl`, `scaler.pkl`
- Run: `python neck_exercise_ml_train.py`

**`neck_exercise_ml_infer.py`** - ML method
- Requires: `trained_model.pkl`, `scaler.pkl`
- Run: `python neck_exercise_ml_infer.py`

## Quick Start

**Option 1 (Geometric):**
```bash
python neck_exercise_geo.py
```

**Option 2 (ML):**
```bash
python neck_exercise_ml_train.py    # Train first
python neck_exercise_ml_infer.py     # Then run
```

## Controls

- `ESC` - Exit
- `D` - Toggle developer mode

## How It Works

Counts reps when head moves: Center → Direction (Up/Down/Left/Right) → Center
- Must hold direction for 30 frames (~1 second)
- Min 0.8 seconds between reps
