"""Prediction-mode regression tests using synthetic HDF5 only.

Run from CHAP2: python -m unittest -v test_prediction_modes
No study data, support files, or trained checkpoints are accessed.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import torch

from chap_model import CHAP
from make_predictions import (
    collect_subject_windows,
    get_args_parser,
    predict_chunks,
    write_subject_csv,
)
from util.commons import input_iterator, window_generator


class PredictionModeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.subject = "33333"
        (self.root / self.subject).mkdir()

    def write_day(self, n=130, sleep=(), nonwear=(), unknown=(),
                  day="2020-01-01", start=0):
        indexes = np.arange(start, start + n)
        data = np.broadcast_to(indexes[:, None, None], (n, 100, 3)).astype(np.float32)
        labels = (indexes % 2).astype(np.int32)
        sleeping = np.zeros(n, dtype=np.int32)
        non_wear = np.zeros(n, dtype=np.int32)
        labels[list(unknown)] = -1
        sleeping[list(sleep)] = 1
        non_wear[list(nonwear)] = 1
        with h5py.File(self.root / self.subject / f"{day}.h5", "w") as fout:
            for name, value in {
                "data": data, "time": indexes * 10.0,
                "label": labels, "sleeping": sleeping, "non_wear": non_wear,
            }.items():
                fout.create_dataset(name, data=value)

    def indexes(self, segments):
        return [x[:, 0, 0].astype(int).tolist() for x, _, _ in segments]

    def collect(self, **kwargs):
        return collect_subject_windows(str(self.root), self.subject, 42, **kwargs)

    def test_default_iterator_preserves_legacy_filtering_and_boundaries(self):
        self.write_day(sleep=(42,), nonwear=(85,), unknown=(10, 100))
        expected = [list(range(42)), list(range(43, 85)), list(range(86, 130))]
        segments = list(input_iterator(str(self.root), self.subject, train=False))
        self.assertEqual(self.indexes(segments), expected)
        self.assertEqual(segments[0][2][10], -1)
        explicit = list(input_iterator(str(self.root), self.subject,
                                       train=False, filter_invalid=True))
        for old, new in zip(segments, explicit):
            for left, right in zip(old, new):
                np.testing.assert_array_equal(left, right)

    def test_consecutive_retains_flags_and_unknown_without_flag_boundaries(self):
        self.write_day(sleep=(42,), nonwear=(85,), unknown=(10, 100))
        segments = list(input_iterator(str(self.root), self.subject,
                                       train=False, filter_invalid=False))
        self.assertEqual(self.indexes(segments), [list(range(130))])
        self.assertEqual(np.count_nonzero(segments[0][2] == -1), 2)

    def test_training_default_still_excludes_all_invalid_windows(self):
        self.write_day(sleep=(42,), nonwear=(85,), unknown=(10, 100))
        segments = list(input_iterator(str(self.root), self.subject, train=True))
        expected = [list(range(10)), list(range(11, 42)), list(range(43, 85)),
                    list(range(86, 100)), list(range(101, 130))]
        self.assertEqual(self.indexes(segments), expected)
        self.assertTrue(all(np.all(y != -1) for _, _, y in segments))
        sequences = list(window_generator(str(self.root), 42, [self.subject]))
        self.assertEqual(len(sequences), 1)
        np.testing.assert_array_equal(sequences[0][0][:, 0, 0], np.arange(43, 85))

    def test_filtered_chunks_do_not_cross_excluded_windows(self):
        self.write_day(sleep=(42,), nonwear=(85,), unknown=(10, 100))
        x, y, ts = self.collect()
        self.assertEqual(x.shape, (3, 42, 100, 3))
        self.assertEqual(y.shape, (3, 42))
        self.assertEqual(ts.shape, (3, 42))
        np.testing.assert_array_equal(x[:, :, 0, 0],
                                      [np.arange(42), np.arange(43, 85), np.arange(86, 128)])

    def test_consecutive_chunks_keep_original_order_and_drop_only_tail(self):
        self.write_day(sleep=(42,), nonwear=(85,), unknown=(10, 100))
        x, y, ts = self.collect(filter_invalid=False)
        self.assertEqual(x.shape, (3, 42, 100, 3))
        np.testing.assert_array_equal(x[:, :, 0, 0].ravel(), np.arange(126))
        np.testing.assert_array_equal(ts.ravel(), np.arange(126) * 10.0)
        self.assertEqual(np.count_nonzero(y == -1), 2)

    def test_modes_match_when_no_sleep_or_nonwear_flags_exist(self):
        self.write_day(n=85, unknown=(10,))
        for filtered, consecutive in zip(self.collect(), self.collect(filter_invalid=False)):
            np.testing.assert_array_equal(filtered, consecutive)

    def test_short_filtered_fragments_not_joined_or_padded(self):
        self.write_day(n=43, sleep=(21,))
        self.assertEqual(self.collect(), (None, None, None))
        self.assertEqual(self.collect(filter_invalid=False)[0].shape, (1, 42, 100, 3))

    def test_ap_absent_unknown_labels_supported_in_both_prediction_modes(self):
        self.write_day(n=84, unknown=tuple(range(84)))
        for filter_invalid in (True, False):
            x, y, _ = self.collect(filter_invalid=filter_invalid)
            self.assertEqual(x.shape, (2, 42, 100, 3))
            self.assertTrue(np.all(y == -1))
        self.assertEqual(list(input_iterator(str(self.root), self.subject, train=True)), [])

    def test_consecutive_dates_keep_existing_cross_day_behavior(self):
        self.write_day(n=30)
        self.write_day(n=54, day="2020-01-02", start=30)
        for filter_invalid in (True, False):
            x, _, _ = self.collect(filter_invalid=filter_invalid)
            np.testing.assert_array_equal(x[:, :, 0, 0].ravel(), np.arange(84))

    def test_missing_calendar_day_still_raises_in_both_modes(self):
        self.write_day(n=42)
        self.write_day(n=42, day="2020-01-03", start=42)
        for filter_invalid in (True, False):
            with self.assertRaises(AssertionError):
                self.collect(filter_invalid=filter_invalid)

    def test_cli_default_and_consecutive_choice(self):
        required = ["--data_dir", "synthetic", "--checkpoint", "synthetic.pth",
                    "--prediction_dir", "synthetic-output"]
        parser = get_args_parser()
        self.assertEqual(parser.parse_args(required).prediction_mode, "filtered")
        args = parser.parse_args(required + ["--prediction_mode", "consecutive"])
        self.assertEqual(args.prediction_mode, "consecutive")

    def test_cpu_prediction_and_csv_still_output_only_binary_predictions(self):
        self.write_day(n=42, sleep=(5,), nonwear=(15,), unknown=(10,))
        x, y, ts = self.collect(filter_invalid=False)
        torch.manual_seed(0)
        model = CHAP(2, 42, 2)
        args = SimpleNamespace(batch_size=1, window_size=42)
        preds = predict_chunks(model, x, args, torch.device("cpu"))
        self.assertEqual(preds.shape, (42,))
        self.assertTrue(set(preds.tolist()).issubset({0, 1}))
        output = self.root / "synthetic_predictions.csv"
        write_subject_csv(output, y, ts, preds)
        frame = pd.read_csv(output)
        self.assertEqual(len(frame), 42)
        self.assertTrue(set(frame.prediction).issubset({"sitting", "not-sitting"}))
        self.assertEqual(frame.label.iloc[10], "no-label")


if __name__ == "__main__":
    unittest.main()
