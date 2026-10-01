"""Header-only tests for SOL/PASOS shard selection."""

import gzip
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

from run_sol_pasos import LABEL_MAP, discover_inputs, run_shard
from summarize_sol_pasos import summarize_location


def write_header_pair(raw_dir, ap_dir, subject, frequency):
    raw_dir.mkdir(parents=True, exist_ok=True)
    ap_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw_dir / f'{subject}.csv.gz', 'wt') as raw:
        raw.write(f'ActiGraph Raw Data; {frequency} Hz\n')
    (ap_dir / f'{subject}.csv').write_text(
        '"TS_LOCAL","PL_ACTIVITY_NEW"\n', encoding='utf-8'
    )


def write_support(support, subjects, location):
    support.mkdir(parents=True)
    (support / 'PASOS_randomization.csv').write_text(
        'concurrentID,randomization\n' +
        ''.join(f'{subject},{split}\n' for subject, split in subjects),
        encoding='utf-8',
    )
    valid_name = ('PASOS_hip_concurrentWear.csv' if location == 'hip'
                  else 'PASOS_concurrentWear.csv')
    nonwear_name = ('VIDA_NW_choi.csv' if location == 'hip'
                    else 'PASOS_NW_choi.csv')
    for name in (valid_name, nonwear_name, 'VIDA_SL.csv'):
        (support / name).write_text('ID\n', encoding='utf-8')


class DiscoveryTests(unittest.TestCase):
    def test_label_map_matches_chap2_default(self):
        self.assertEqual(LABEL_MAP, {'0': 0, '1': 1, '2': 1})

    def test_hip_30_and_80_hz_use_randomization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'SOL' / 'hip_data' / 'PASOS_hip'
            write_support(root / 'support_files',
                          [('H001', 'train'), ('H002', 'test')], 'hip')
            write_header_pair(root / 'AG_RAW', root / 'AP', 'H001', 30)
            write_header_pair(root / 'AG_RAW', root / 'AP', 'H002', 80)
            records, *_ = discover_inputs(Path(temporary) / 'SOL', 'hip')
            self.assertEqual({(split, hz) for split, hz, *_ in records},
                             {('train', 30), ('test', 80)})

    def test_wrist_60_and_80_hz_preserve_source_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'SOL' / 'PASOS'
            write_support(root / 'PASOS_support_files',
                          [('W001', 'train'), ('W002', 'test')], 'wrist')
            write_header_pair(root / 'train' / 'AG_RAW', root / 'train' / 'AP',
                              'W001', 60)
            write_header_pair(root / 'test' / 'AG_RAW', root / 'test' / 'AP',
                              'W002', 80)
            records, *_ = discover_inputs(Path(temporary) / 'SOL', 'wrist')
            self.assertEqual({(split, hz) for split, hz, *_ in records},
                             {('train', 60), ('test', 80)})

    def test_rejects_source_split_conflict(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'SOL' / 'PASOS'
            write_support(root / 'PASOS_support_files',
                          [('W001', 'test')], 'wrist')
            write_header_pair(root / 'train' / 'AG_RAW', root / 'train' / 'AP',
                              'W001', 60)
            (root / 'test' / 'AG_RAW').mkdir(parents=True)
            (root / 'test' / 'AP').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'wrong source'):
                discover_inputs(Path(temporary) / 'SOL', 'wrist')

    def test_shards_write_to_one_location_subject_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'SOL' / 'PASOS'
            write_support(root / 'PASOS_support_files',
                          [('W001', 'train'), ('W002', 'test')], 'wrist')
            for split, subject in (('train', 'W001'), ('test', 'W002')):
                write_header_pair(root / split / 'AG_RAW', root / split / 'AP',
                                  subject, 60)
            args = SimpleNamespace(
                source_root=Path(temporary) / 'SOL',
                output_root=Path(temporary) / 'output',
                location='wrist', frequency=60,
                scratch_root=Path(temporary), workers=1, dry_run=False,
            )
            with patch('run_sol_pasos.subprocess.run') as run:
                run_shard(args)
            self.assertEqual(run.call_count, 2)
            for call in run.call_args_list:
                command = call.args[0]
                index = command.index('--pre-processed-dir')
                self.assertEqual(command[index + 1],
                                 str(args.output_root / 'processed' / 'wrist'))

    def test_aggregate_report_counts_only_eligible_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            source_root = Path(temporary) / 'SOL'
            support = source_root / 'PASOS' / 'PASOS_support_files'
            write_support(support, [('W001', 'train'), ('W002', 'test')], 'wrist')
            (support / 'train_val_split.csv').write_text(
                'subject_id,split\nW001,validation\nW002,test\n', encoding='utf-8'
            )
            output_root = Path(temporary) / 'output'
            for source_split, subject, labels, sleeping, nonwear in (
                ('train', 'W001', [0, 1, -1, 0], [0, 0, 0, 1], [0, 0, 0, 0]),
                ('test', 'W002', [0, 1], [0, 0], [0, 1]),
            ):
                folder = output_root / 'processed' / 'wrist' / subject
                folder.mkdir(parents=True)
                with h5py.File(folder / '2026-01-01.h5', 'w') as destination:
                    destination.create_dataset('label', data=np.array(labels))
                    destination.create_dataset('sleeping', data=np.array(sleeping))
                    destination.create_dataset('non_wear', data=np.array(nonwear))
            rows, extras = summarize_location(source_root, output_root, 'wrist')
            by_partition = {row['partition']: row for row in rows}
            self.assertEqual(extras, 0)
            self.assertEqual(by_partition['development']['all_10s_windows'], 4)
            self.assertEqual(by_partition['validation']['usable_10s_windows'], 2)
            self.assertEqual(by_partition['test']['sitting_windows'], 1)
            self.assertEqual(by_partition['test']['non_sitting_windows'], 0)


if __name__ == '__main__':
    unittest.main()
