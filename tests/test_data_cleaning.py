"""Run with: python -m unittest discover -s tests"""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_cleaning import clean_data


class CleaningCheck(unittest.TestCase):
    def test_cleaning_contract(self):
        # All required keys are constant in this small dataset: never drop them.
        row = dict(RENT_APPROVAL_DATE="2025-03", TOWN=" Town  ", BLOCK="001a",
                   STREET="A  Street", FLAT_TYPE="4 ROOM", FLAT_MODEL="Model A",
                   FLOOR_AREA_SQM=90, FURNISHED="yes", LEASE_COMMENCE_DATE=2000,
                   FEE=0, MONTHLY_RENT=2700.5)
        raw = pd.DataFrame([row, row])
        raw_test = raw.drop(columns="MONTHLY_RENT")
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            p = Path(tmp)
            raw.to_csv(p / "train.csv", index=False)
            raw_test.to_csv(p / "test.csv", index=False)
            tr, te = clean_data(p / "train.csv", p / "test.csv", output_dir=p / "out")
            self.assertEqual(len(tr), 2)
            self.assertEqual(len(te), 2)
            self.assertEqual(tr.MONTHLY_RENT.tolist(), [2700.5, 2700.5])
            self.assertEqual(te.BLOCK.tolist(), ["001A", "001A"])
            self.assertEqual(te.FLAT_TYPE.tolist(), ["4-room", "4-room"])
            self.assertEqual(list(te), list(tr.drop(columns="MONTHLY_RENT")))
            self.assertTrue((p / "out" / "processed_test.csv").exists())
            dedup, _ = clean_data(p / "train.csv", p / "test.csv", write=False, drop_exact_duplicates=True)
            self.assertEqual(len(dedup), 1)
            for column, value in [("FEE", 1), ("FLAT_TYPE", "  "),
                                  ("RENT_APPROVAL_DATE", "2025-13"),
                                  ("FLOOR_AREA_SQM", -1), ("FLOOR_AREA_SQM", float("inf")),
                                  ("LEASE_COMMENCE_DATE", 2000.5)]:
                altered = raw_test.copy()
                altered[column] = value
                altered.to_csv(p / "test.csv", index=False)
                with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                    clean_data(p / "train.csv", p / "test.csv", write=False)


if __name__ == "__main__":
    unittest.main()
