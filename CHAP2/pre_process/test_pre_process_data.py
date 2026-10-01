"""Synthetic checks for the CHAP2 one-second activPAL preprocessing path."""

import io
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import h5py
import pandas as pd

import pre_process_data as preprocessing


def synthetic_raw():
    header = [
        'header', 'serial', 'Start Time 12:00:00', 'Start Date 1/1/2023',
    ] + ['metadata'] * 7
    return io.StringIO('\n'.join(header + ['0,0,1'] * 100) + '\n')


def synthetic_labels():
    start = datetime(2023, 1, 1, 12)
    return pd.DataFrame({
        'TS_LOCAL': [
            (start + timedelta(seconds=i)).strftime('%Y-%m-%dT%H:%M:%SZ')
            for i in range(10)
        ],
        'TS_LOCAL_COR': [
            (start + timedelta(days=1, seconds=i)).strftime('%Y-%m-%dT%H:%M:%SZ')
            for i in range(10)
        ],
        'PL_ACTIVITY_NEW': [1] * 10,
    })


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        preprocessing.args = SimpleNamespace(
            down_sample_frequency=10,
            gt3x_frequency=10,
            window_size=10,
            event_file=False,
            silent=True,
        )

    def test_epoch_labels_follow_ts_local_and_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as output:
            preprocessing.map_function(
                synthetic_raw(), {}, {}, {}, {}, output, 'synthetic',
                synthetic_labels(), {'1': 1},
            )
            with h5py.File(os.path.join(output, 'synthetic', '2023-01-01.h5')) as result:
                self.assertEqual(result['data'].shape, (1, 100, 3))
                self.assertEqual(result['label'][:].tolist(), [1])

            with self.assertRaises(FileExistsError):
                preprocessing.map_function(
                    synthetic_raw(), {}, {}, {}, {}, output, 'synthetic',
                    synthetic_labels(), {'1': 1},
                )

    def test_recording_failure_preserves_existing_subject_output(self):
        with tempfile.TemporaryDirectory() as root:
            output = os.path.join(root, 'output')
            subject_dir = os.path.join(output, 'synthetic')
            os.makedirs(subject_dir)
            marker = os.path.join(subject_dir, 'existing.txt')
            with open(marker, 'w', encoding='utf-8') as handle:
                handle.write('keep')

            with self.assertRaises(FileNotFoundError):
                preprocessing.fn(
                    'synthetic', 'missing', {}, preprocessing.args, {}, {}, None,
                    {}, output, root, False, '.csv',
                )
            self.assertTrue(os.path.isfile(marker))

    def test_sol_support_file_headers_and_timestamps(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            raw_dir = root / 'raw'
            raw_dir.mkdir()
            sleep = root / 'sleep.csv'
            sleep.write_text(
                'ID,startSL,endSL\n'
                'synthetic,2023-01-01 22:00:00,2023-01-02 08:00:00\n',
                encoding='utf-8',
            )
            non_wear = root / 'non_wear.csv'
            non_wear.write_text(
                'ID,startNW,endNW\n'
                'synthetic,2023-01-01 12:00:00,2023-01-01 12:10:00\n',
                encoding='utf-8',
            )
            preprocessing.generate_pre_processed_data(
                str(raw_dir), None, {'0': 0, '1': 1, '2': 1},
                sleep_logs_file=str(sleep),
                non_wear_times_file=str(non_wear),
                pre_process_data_output_dir=str(root / 'output'),
                gzipped=True,
            )


if __name__ == '__main__':
    unittest.main()
