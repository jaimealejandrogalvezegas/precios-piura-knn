import unittest

from src.config import CLIMATE_FILE, FEATURES_B, MARKET_FILE, NINO_FILE, SERIES_KEYS
from src.data_processing import (
    consolidate_sources,
    load_raw_sources,
    model_ready_rows,
    quality_report,
)


class DataPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sources = load_raw_sources(MARKET_FILE, CLIMATE_FILE, NINO_FILE)
        cls.consolidated = consolidate_sources(*sources)
        cls.ready = model_ready_rows(cls.consolidated)

    def test_expected_volume(self):
        self.assertEqual(len(self.consolidated), 14911)
        self.assertEqual(len(self.ready), 14605)
        self.assertEqual(self.ready["PRODUCTO"].nunique(), 51)

    def test_no_duplicate_series_date(self):
        duplicates = self.consolidated.duplicated(SERIES_KEYS + ["FECHA"]).sum()
        self.assertEqual(int(duplicates), 0)

    def test_complete_model_features(self):
        self.assertEqual(int(self.ready[FEATURES_B].isna().sum().sum()), 0)

    def test_temporal_guards(self):
        report = quality_report(self.consolidated)
        self.assertEqual(report["violaciones_clima_futuro"], 0)
        self.assertEqual(report["violaciones_nino_desfase"], 0)


if __name__ == "__main__":
    unittest.main()

