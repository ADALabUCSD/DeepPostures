"""Run one SOL/PASOS wear-location and raw-frequency preprocessing shard.

Input selection uses filenames, ActiGraph metadata headers, activPAL headers,
and the study-provided randomization file. It does not inspect data rows before
the normal preprocessing step. Files are linked into temporary flat directories
so the original CHAP2 preprocessing CLI can operate on one frequency at a time.
"""

import argparse
import csv
import gzip
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


LABEL_MAP = {'0': 0, '1': 1, '2': 1}


def read_randomization(path):
    with path.open(encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source)
        if not {'concurrentID', 'randomization'} <= set(reader.fieldnames or []):
            raise ValueError(f'Unsupported randomization header: {path}')
        splits = {}
        for row in reader:
            subject = row['concurrentID'].strip()
            split = row['randomization'].strip().lower()
            if split not in {'train', 'test'}:
                raise ValueError(f'Unexpected randomization split in {path}')
            if subject in splits and splits[subject] != split:
                raise ValueError(f'Conflicting randomization assignments in {path}')
            splits[subject] = split
    return splits


def raw_frequency(path):
    with gzip.open(path, 'rt') as source:
        first_line = source.readline()
    match = re.search(r'\b(\d+)\s*Hz\b', first_line, re.IGNORECASE)
    if not match:
        raise ValueError('ActiGraph frequency is absent from a raw-file header')
    return int(match.group(1))


def discover_inputs(source_root, location):
    if location == 'wrist':
        root = source_root / 'PASOS'
        support = root / 'PASOS_support_files'
        valid_days = support / 'PASOS_concurrentWear.csv'
        non_wear = support / 'PASOS_NW_choi.csv'
        source_pairs = [(split, root / split / 'AG_RAW', root / split / 'AP')
                        for split in ('train', 'test')]
    else:
        root = source_root / 'hip_data' / 'PASOS_hip'
        support = root / 'support_files'
        valid_days = support / 'PASOS_hip_concurrentWear.csv'
        non_wear = support / 'VIDA_NW_choi.csv'
        source_pairs = [(None, root / 'AG_RAW', root / 'AP')]

    randomization = read_randomization(support / 'PASOS_randomization.csv')
    sleep = support / 'VIDA_SL.csv'
    for path in (valid_days, non_wear, sleep):
        if not path.is_file():
            raise FileNotFoundError(f'Required support file is missing: {path}')

    records = []
    seen_subjects = set()
    for source_split, raw_dir, ap_dir in source_pairs:
        if not raw_dir.is_dir() or not ap_dir.is_dir():
            raise FileNotFoundError('Raw or activPAL input directory is missing')
        raw_paths = sorted(raw_dir.glob('*.csv.gz'))
        ap_stems = {path.stem for path in ap_dir.glob('*.csv')}
        raw_stems = {path.name[:-7] for path in raw_paths}
        if raw_stems != ap_stems:
            raise ValueError('Raw and activPAL basenames do not match')
        for raw_path in raw_paths:
            subject = raw_path.name[:-7]
            if subject in seen_subjects:
                raise ValueError('A subject has multiple raw recordings in this location')
            seen_subjects.add(subject)
            if subject not in randomization:
                raise ValueError('A raw-file subject is absent from the randomization list')
            assigned_split = randomization[subject]
            if source_split is not None and source_split != assigned_split:
                raise ValueError('A raw file is in the wrong source train/test directory')
            frequency = raw_frequency(raw_path)
            supported = {30, 80} if location == 'hip' else {60, 80}
            if frequency not in supported:
                raise ValueError(f'Unsupported raw sampling frequency: {frequency} Hz')
            ap_path = ap_dir / f'{subject}.csv'
            with ap_path.open(encoding='utf-8-sig', newline='') as source:
                header = next(csv.reader(source))
            if 'TS_LOCAL' not in header or 'PL_ACTIVITY_NEW' not in header:
                raise ValueError('An activPAL file lacks TS_LOCAL or PL_ACTIVITY_NEW')
            records.append((assigned_split, frequency, subject, raw_path, ap_path))

    if seen_subjects != set(randomization):
        raise ValueError('Randomization list and raw subject set differ')
    return records, valid_days, sleep, non_wear


def run_shard(args):
    records, valid_days, sleep, non_wear = discover_inputs(args.source_root, args.location)
    counts = Counter((split, hz) for split, hz, *_ in records)
    print('INPUT_COUNTS', args.location, dict(sorted(counts.items())), flush=True)
    selected = [record for record in records if record[1] == args.frequency]
    print('SELECTED_RECORDINGS', args.location, args.frequency, len(selected), flush=True)
    if args.dry_run:
        return
    if not selected:
        raise ValueError('No recordings match the requested shard')

    output_dir = args.output_root / 'processed' / args.location
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'test'):
        subset = [record for record in selected if record[0] == split]
        if not subset:
            continue
        for _, _, subject, _, _ in subset:
            if (output_dir / subject).exists():
                raise FileExistsError('Output for a selected subject already exists')

        with tempfile.TemporaryDirectory(dir=args.scratch_root) as temporary:
            raw_stage = Path(temporary) / 'raw'
            ap_stage = Path(temporary) / 'ap'
            raw_stage.mkdir()
            ap_stage.mkdir()
            for _, _, subject, raw_path, ap_path in subset:
                (raw_stage / f'{subject}.csv.gz').symlink_to(raw_path)
                (ap_stage / f'{subject}.csv').symlink_to(ap_path)
            command = [
                sys.executable, str(Path(__file__).with_name('pre_process_data.py')),
                '--gt3x-dir', str(raw_stage),
                '--activpal-dir', str(ap_stage),
                '--pre-processed-dir', str(output_dir),
                '--valid-days-file', str(valid_days),
                '--sleep-logs-file', str(sleep),
                '--non-wear-times-file', str(non_wear),
                '--gt3x-frequency', str(args.frequency),
                '--down-sample-frequency', '10',
                '--window-size', '10',
                '--activpal-label-map', json.dumps(LABEL_MAP),
                '--mp', str(args.workers),
                '--gzipped', '--silent',
            ]
            print('PROCESSING', args.location, args.frequency, split,
                  'recordings', len(subset), flush=True)
            subprocess.run(command, check=True)
            print('COMPLETED', args.location, args.frequency, split, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path,
                        default=Path('/niddk-data-central/SOL'))
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--location', choices=('hip', 'wrist'), required=True)
    parser.add_argument('--frequency', choices=(30, 60, 80), type=int, required=True)
    parser.add_argument('--scratch-root', type=Path, default=Path('/tmp'))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be at least 1')
    if (args.location, args.frequency) not in {
        ('hip', 30), ('hip', 80), ('wrist', 60), ('wrist', 80)
    }:
        parser.error('Unsupported location/frequency pair')
    run_shard(args)


if __name__ == '__main__':
    main()
