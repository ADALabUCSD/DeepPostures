"""Synthetic checks for the CHAP2 one-second activPAL preprocessing path."""

import io
import os
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd

import pre_process_data as preprocessing


def synthetic_raw(start_date='1/1/2023', start_time='12:00:00'):
    header = [
        'header', 'serial', f'Start Time {start_time}', f'Start Date {start_date}',
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
        preprocessing.label_map = {'0': 0, '1': 1, '2': 1}

    def parse_nonwear(self, content, loc=None):
        """Capture parsed intervals before processing any raw recording."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / 'raw'
            raw.mkdir()
            (raw / 'synthetic.csv').touch()
            logs = root / 'nonwear.csv'
            logs.write_text(content, encoding='utf-8')
            with patch.object(preprocessing, 'fn') as process:
                preprocessing.generate_pre_processed_data(
                    str(raw), None, preprocessing.label_map,
                    non_wear_times_file=str(logs), loc=loc,
                    pre_process_data_output_dir=str(root / 'output'),
                )
            return process.call_args.args[2]

    def test_iwatch_filters_loc_column_with_original_minute_aligned_times(self):
        content = (
            'ID,wearLoc,NW_DT,int.min,wearDate,timeNum,filename,loc\n'
            'synthetic,H,2026-11-05 09:09:00,5,2026-11-05,0,example,Wrist\n'
            'synthetic,W,2026-11-05 10:09:00,2,2026-11-05,0,example,Hip\n'
        )
        wrist = self.parse_nonwear(content, 'wrist')['synthetic']
        hip = self.parse_nonwear(content, 'hip')['synthetic']
        self.assertEqual(wrist, [(datetime(2026, 11, 5, 9, 9),
                                  datetime(2026, 11, 5, 9, 14))])
        self.assertEqual(hip, [(datetime(2026, 11, 5, 10, 9),
                                datetime(2026, 11, 5, 10, 11))])

    def test_iwatch_schema_requires_explicit_location_even_for_one_location(self):
        with self.assertRaisesRegex(ValueError, '--loc must be provided'):
            self.parse_nonwear(
                'ID,wearLoc,NW_DT,int.min,wearDate,timeNum,filename,loc\n'
                'synthetic,W,2026-11-05,1440,2026-11-05,0,example,Wrist\n'
            )

    def test_wearloc_alone_is_not_a_supported_schema(self):
        with self.assertRaisesRegex(Exception, 'Unsupported non_wear_times_file format'):
            self.parse_nonwear(
                'ID,wearLoc,startNW,endNW\n'
                'synthetic,W,2026-11-05 09:09:00,2026-11-05 09:14:30\n',
                'wrist',
            )

    def test_loc_does_not_filter_sol_records_and_warns(self):
        preprocessing.args.silent = False
        with self.assertLogs(preprocessing.logger, level='WARNING') as messages:
            result = self.parse_nonwear(
                'ID,startNW,endNW\n'
                'synthetic,2026-11-05 09:09:15,2026-11-05 09:14:30\n',
                'hip',
            )
        self.assertEqual(result['synthetic'][0][0].second, 15)
        self.assertTrue(any('--loc has no effect' in message for message in messages.output))

    def test_loc_does_not_filter_chap1_records(self):
        result = self.parse_nonwear(
            'ID,Date.nw.start,Time.nw.start,Date.nw.end,Time.nw.end\n'
            'synthetic,11/05/2026,09:09,11/05/2026,09:14\n', 'wrist',
        )
        self.assertEqual(result['synthetic'], [(datetime(2026, 11, 5, 9, 9),
                                               datetime(2026, 11, 5, 9, 14))])

    def test_iwatch_event_labels_preserve_binary_and_unknown_classes(self):
        preprocessing.args.event_file = True
        start = datetime(2023, 1, 1, 12)
        serial = (start - datetime(1970, 1, 1)).total_seconds() / 86400 + 25569
        for code, expected in ((0, 0), (1, 1), (-1, -1), (-2, -1)):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as output:
                events = pd.DataFrame({
                    'Time': [serial], 'Interval (s)': [10.0],
                    'ActivityCode (0=sedentary, 1= standing, 2=stepping)': [code],
                })
                preprocessing.map_function(
                    synthetic_raw(), {}, {}, {}, {}, output, 'synthetic',
                    events, preprocessing.label_map,
                )
                with h5py.File(Path(output) / 'synthetic/2023-01-01.h5', 'r') as result:
                    self.assertEqual(result['label'][:].tolist(), [expected])


    def test_sol_rejects_reversed_nonwear_interval(self):
        with self.assertRaisesRegex(ValueError, 'end must not precede'):
            self.parse_nonwear(
                'ID,startNW,endNW\n'
                'synthetic,2026-11-05 09:14:30,2026-11-05 09:09:15\n'
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

    def test_activity_codes_accept_numeric_csv_types_and_exclude_unknowns(self):
        label_map = {'0': 0, '1': 1, '2': 1}
        for value, expected in (
            (0, 0), (0.0, 0), ('0.0', 0),
            (1, 1), (1.0, 1), (2.0, 1),
            (float('nan'), -1), ('', -1), (3, -1), (-1, -1),
        ):
            with self.subTest(value=value):
                self.assertEqual(preprocessing.map_activity_code(value, label_map), expected)

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
        results = []
        cases = (
            ('startsleep', 'endsleep', '8/26/26 8:59', '8/26/26 9:00'),
            ('startSL', 'endSL', '2026-08-26 8:59:00', '2026-08-26 9:00:00'),
        )
        for start_header, end_header, start, end in cases:
            with self.subTest(start_header=start_header), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                raw_dir = root / 'raw'
                raw_dir.mkdir()
                (raw_dir / 'synthetic.csv').write_text(
                    synthetic_raw('8/26/2026', '08:59:00').getvalue(), encoding='utf-8'
                )
                sleep = root / 'sleep.csv'
                sleep.write_text(
                    f'ID,{start_header},{end_header}\nsynthetic,{start},{end}\n',
                    encoding='utf-8',
                )
                non_wear = root / 'non_wear.csv'
                non_wear.write_text(
                    f'ID,startNW,endNW\nsynthetic,{start},{end}\n',
                    encoding='utf-8',
                )
                preprocessing.generate_pre_processed_data(
                    str(raw_dir), None, {'0': 0, '1': 1, '2': 1},
                    sleep_logs_file=str(sleep),
                    non_wear_times_file=str(non_wear),
                    pre_process_data_output_dir=str(root / 'output'),
                )
                with h5py.File(root / 'output' / 'synthetic' / '2026-08-26.h5') as output:
                    self.assertEqual(output['data'].shape, (1, 100, 3))
                    self.assertEqual(output['sleeping'][:].tolist(), [1])
                    self.assertEqual(output['non_wear'][:].tolist(), [1])
                    results.append({name: output[name][:] for name in output})
        self.assertEqual(results[0].keys(), results[1].keys())
        for name in results[0]:
            np.testing.assert_array_equal(results[0][name], results[1][name])


if __name__ == '__main__':
    unittest.main()
