import json
import unittest

import joblib

from src.config import FEATURES_B, METADATA_FILE, MODEL_B_FILE, PROCESSED_FILE
from src.data_processing import load_processed, prediction_ready_rows
from src.prediction import predict_row


class ModelArtifactsTest(unittest.TestCase):
    def test_artifacts_exist_and_predict(self):
        self.assertTrue(PROCESSED_FILE.exists())
        self.assertTrue(MODEL_B_FILE.exists())
        self.assertTrue(METADATA_FILE.exists())

        data = load_processed(PROCESSED_FILE)
        model = joblib.load(MODEL_B_FILE)
        metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        row = prediction_ready_rows(data).tail(1)
        label, support = predict_row(model, row)

        self.assertIn(label, metadata["clases"])
        self.assertAlmostEqual(sum(support.values()), 1.0, places=6)
        self.assertEqual(int(row[FEATURES_B].isna().sum().sum()), 0)


if __name__ == "__main__":
    unittest.main()
