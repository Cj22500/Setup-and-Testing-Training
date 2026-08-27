import argparse
import json
import os
import joblib
import heatmap_feature_extractor as hfe

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "suspicion_model.joblib")
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, "feature_columns.json")


def predict_session(session_dir):
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    features = hfe.extract_session_features(session_dir)
    if features is None:
        print(f"[Predictor] Session missing heatmap or CSV log: {session_dir}")
        return None

    import pandas as pd
    ordered_df = pd.DataFrame([[features.get(col, 0.0) for col in feature_columns]], columns=feature_columns)
    probability_cheating = model.predict_proba(ordered_df)[0][1]

    label = "cheating" if probability_cheating >= 0.5 else "non_cheating"
    confidence = probability_cheating if label == "cheating" else 1 - probability_cheating
    print(f"[Predictor] {os.path.basename(session_dir)} -> {label} ({confidence * 100:.1f}% confidence)")
    return label, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict cheating/non_cheating for a single session.")
    parser.add_argument("session_dir", help="Path to a session directory containing a heatmap PNG and CSV log.")
    args = parser.parse_args()
    predict_session(args.session_dir)
