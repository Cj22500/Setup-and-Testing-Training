import os
import shutil
import tempfile
import unittest
import numpy as np
import cv2
import pandas as pd
import heatmap_feature_extractor as hfe
import dataset_builder
import train_model
import predict_session


class TestMLPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.session_dir = os.path.join(self.temp_dir, "session_test")
        os.makedirs(self.session_dir, exist_ok=True)

        # Create dummy heatmap PNG
        heatmap_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(heatmap_img, (50, 50), 20, (0, 0, 255), -1)
        self.heatmap_path = os.path.join(self.session_dir, "session_heatmap_12345.png")
        cv2.imwrite(self.heatmap_path, heatmap_img)

        # Create dummy CSV log
        self.csv_path = os.path.join(self.session_dir, "session_log_12345.csv")
        csv_content = (
            "Timestamp start,Timestamp finish,Gaze direction,Violation label\n"
            "2026-01-01 10:00:00.000,2026-01-01 10:00:02.000,Left,frantic_eye_movement_violation\n"
            "2026-01-01 10:00:02.000,2026-01-01 10:00:05.000,Center,normal\n"
            "2026-01-01 10:00:05.000,2026-01-01 10:00:08.000,Down,off_screen_violation\n"
        )
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_recover_intensity_map(self):
        img = cv2.imread(self.heatmap_path)
        intensity = hfe.recover_intensity_map(img)
        self.assertEqual(intensity.shape, (100, 100))
        self.assertGreater(np.max(intensity), 0)

    def test_extract_session_features(self):
        features = hfe.extract_session_features(self.session_dir)
        self.assertIsNotNone(features)
        self.assertIn("centroid_x_norm", features)
        self.assertIn("violation_count_frantic_eye_movement", features)
        self.assertIn("violation_count_off_screen", features)
        self.assertEqual(features["num_transitions"], 3)

    def test_train_and_predict(self):
        # Verify model training with existing features.csv if present
        if os.path.exists("features.csv"):
            df = pd.read_csv("features.csv")
            self.assertGreater(len(df), 0)
            train_model.train_model()
            self.assertTrue(os.path.exists(train_model.MODEL_PATH))
            self.assertTrue(os.path.exists(train_model.FEATURE_COLUMNS_PATH))

            # Test prediction on our dummy session
            res = predict_session.predict_session(self.session_dir)
            self.assertIsNotNone(res)
            label, confidence = res
            self.assertIn(label, ("cheating", "non_cheating"))
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
