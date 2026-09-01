# MAKITEST System Architecture & Technical Investigation Report

## Executive Summary

This document provides a comprehensive technical overview of the **MAKITEST Suspicious Exam Behaviour Detection System**. MAKITEST is an automated proctoring platform that combines real-time computer vision (facial landmarking and eye-gaze tracking) with machine learning behavior classification (Calibrated Random Forest) to monitor exam takers, log suspicious activities, generate visual gaze heatmaps, record incident video clips, and predict overall cheating probabilities.

---

## 1. System Architecture & Workflow

The system operates across two main pipelines:

### 1.1 Real-Time Tracking & Data Collection Pipeline (`main_trackerprocess.py`)
```
Webcam Feed → FrameBufferProcessor (Threaded Buffer)
                   ↓
        FaceTrackProcessor → FaceDistanceProcessor (Iris Width Polynomial Fit)
                   ↓                   ↓
        FaceAxisProcessor       EyeTrackProcessor (MediaPipe 3D Landmark Mesh)
                   ↓                   ↓
        GazeDirectionProcessor ← EyeGazeProcessor & EyeCalibrationProcessor
                   ↓                   ↓
        EyeScreenPosProcessor → Weighted Screen Position Grid (1920x1080)
                   ↓
        SuspicionScoringProcessor + KeypressTrackProcessor (Global Hooks)
                   ↓
   ─────────────────────────────────────────────────────────
  │  Session Outputs Saved to sessions/<category>/session_<id>/ │
  │  1. session_log_<timestamp>.csv (Event Log)              │
  │  2. session_heatmap_<timestamp>.png (Color-mapped PNG)    │
  │  3. clip_<timestamp>.avi (Incident Video Clips)           │
   ─────────────────────────────────────────────────────────
```

### 1.2 Offline Machine Learning & Feature Extraction Pipeline
```
Saved Session Directory (Heatmap PNG + CSV Log)
                   ↓
    heatmap_feature_extractor.py
    ├── Reconstruct 2D Intensity Map via cKDTree JET LUT Inversion
    ├── Extract 72 Spatial Heatmap Features (Centroid, Spread, Entropy, 8x8 Grid)
    └── Extract 7 Behavioral Features (Violation Counts, Transitions, % Non-Center)
                   ↓
         dataset_builder.py → features.csv (97 Sessions, 79 Predictor Columns)
                   ↓
          train_model.py → CalibratedClassifierCV(RandomForestClassifier)
                   ↓
  ───────────────────────────────────────────────────────────
 │ Saved Artifacts:                                          │
 │ - suspicion_model.joblib (Trained Classifier Model)       │
 │ - feature_columns.json   (Feature Schema Alignment)       │
  ───────────────────────────────────────────────────────────
                   ↓
  predict_session.py / Dashboard → Cheating Probability & Confidence Score
```

---

## 2. Complete Technology Stack

| Domain | Technology / Library | Version | Description & Role |
|---|---|---|---|
| **Core Runtime** | **Python** | `3.10+ / 3.12` | Main application programming language |
| **Landmark Tracking** | **Google MediaPipe** | `mediapipe==0.10.35` | 3D Face Landmarker (`face_landmarker.task`) extracting 478 facial mesh coordinates & iris boundaries |
| **Computer Vision** | **OpenCV** | `opencv-python==4.13.0` | Frame acquisition, landmark visualization, colormap rendering (`COLORMAP_JET`), and video recording |
| **Machine Learning** | **Scikit-Learn** | `scikit-learn>=1.5.2` | Calibrated Random Forest Classifier (`CalibratedClassifierCV`, `RandomForestClassifier`), Stratified K-Fold CV |
| **Spatial Analysis** | **SciPy** | `scipy>=1.15` | `cKDTree` for nearest-neighbor colormap inversion & `np.linalg.eigvalsh` for elongation ratio analysis |
| **Data Processing** | **Pandas & NumPy** | `pandas>=2.2.3`, `numpy==2.4.4` | Matrix computations, moment analysis, tabular dataset manipulation, and schema preservation |
| **Model Persistence** | **Joblib** | `joblib>=1.5.3` | Model serialization and loading (`suspicion_model.joblib`) |
| **Keyboard Hooks** | **Keyboard** | `keyboard==0.13.5` | Global low-level hotkey hook monitoring for shortcut evasions (`Alt+Tab`, `Ctrl+C/V`, `Win+Shift+S`) |
| **Visualization** | **Matplotlib & Pillow** | `matplotlib==3.10.9`, `pillow==12.2.0` | Heatmap plotting, Gaussian blurring, and image I/O |
| **Testing & CI/CD** | **GitHub Actions & Unittest** | `.github/workflows/ci.yml` | Automated cloud continuous integration, system library configuration, unit testing (`test_pipeline.py`) |

---

## 3. Detailed Component Investigation

### 3.1 Landmark & Gaze Tracking Modules
- **`face_trackprocessor.py`**: Executes MediaPipe face landmarker to return 478 3D landmarks. Computes a smoothed head normal vector.
- **`face_distanceprocessor.py`**: Fits a 2nd-degree polynomial (`polyfit`) to iris pixel distance to estimate physical camera-to-subject distance in centimeters.
- **`face_axisprocessor.py`**: Converts chin, nose, forehead, and cheek landmarks into yaw and pitch degrees. Applies anchor offsets for off-center face positions.
- **`eye_trackprocessor.py` & `eye_gazeprocessor.py`**: Isolates eye bounding boxes, eyelid heights, and normalized horizontal iris coordinates.
- **`eye_calibrationprocessor.py`**: Manages a 5-stage calibration sequence (Center, Up, Down, Left, Right) with 60 samples per stage stored in `calibration_data.json`.
- **`eye_screenposprocessor.py` & `gaze_directionprocessor.py`**: Fuses face axis (45%) and eye gaze (55%) estimates with Exponential Moving Average (EMA) smoothing (`tau = 0.18s`) into a 3x3 screen grid.

### 3.2 Violation & Heatmap Processors
- **`suspicion_scoringprocessor.py`**: Monitors threshold triggers (Side looking > 3.0s, Down looking > 5.0s, Off-screen > 1.5s, Frantic shifts > 6 / 10s) and triggers AVI video clip saves.
- **`keypress_trackprocessor.py`**: Listens for evasive key combos (`Alt+Tab`, `Ctrl+C/V/X`, `Win+Shift+S`, `Alt+F4`, `Ctrl+Esc`).
- **`heatmap_processor.py`**: Accumulates gaze positions, applies Gaussian blurring, and saves a JET color-mapped PNG on session exit.

### 3.3 Feature Extraction & Machine Learning Pipeline
- **`heatmap_feature_extractor.py`**:
  - Uses `cKDTree` on a 256-entry `COLORMAP_JET` lookup table to invert session heatmaps into quantitative 2D density maps.
  - Computes spatial features: normalized centroids $(x, y)$, standard deviation spreads $(\sigma_x, \sigma_y)$, covariance eigenvalue elongation ratio, Shannon entropy on an $8\times 8$ grid, peak concentration ratio, total screen coverage ratio, and 64 grid cell probabilities (`grid_cell_0` to `63`).
  - Computes behavioral features: violation counts per category, total gaze transitions, percentage of non-center time, and overall violation rate.
- **`train_model.py`**: Trains `CalibratedClassifierCV(RandomForestClassifier(n_estimators=200, max_depth=6))` with 5-fold Stratified CV.
- **Model Results**:
  - **Cross-validated Accuracy**: `94.8%`
  - **ROC-AUC**: `0.980`
  - **Precision / Recall / F1**: `0.95` across both `cheating` and `non_cheating` classes.
- **`predict_session.py`**: Loads `suspicion_model.joblib` and `feature_columns.json` to predict cheating probabilities on single session folders.

---

## 4. Continuous Integration & Verification Setup

- **Unit Testing (`test_pipeline.py`)**: Covers colormap intensity recovery, feature extraction from synthetic session data, model retraining, and schema-aligned prediction.
- **GitHub Actions Workflow (`.github/workflows/ci.yml`)**:
  - Runs on `ubuntu-latest` with Python 3.12 on every push/PR to `main`/`master`.
  - Installs required Linux system libraries (`libportaudio2`, `libgl1-mesa-glx`, `libglib2.0-0`).
  - Executes unit test suite (`python -m unittest test_pipeline.py`) and model validation (`python train_model.py`).
