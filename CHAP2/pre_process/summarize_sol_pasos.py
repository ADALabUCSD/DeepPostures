"""Print and save aggregate SOL/PASOS preprocessing counts (never subject rows)."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from run_sol_pasos import read_randomization


def read_train_validation(path):
    with path.open(encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source)
        if not {'subject_id', 'split'} <= set(reader.fieldnames or []):
            raise ValueError('Unsupported train/validation split header')
        splits = {}
        for row in reader:
            subject = row['subject_id'].strip()
            split = row['split'].strip().lower()
            if split not in {'train', 'validation', 'test'}:
                raise ValueError('Unexpected train/validation split value')
            if subject in splits and splits[subject] != split:
                raise ValueError('Conflicting train/validation split')
            splits[subject] = split
    return splits


def summarize_location(source_root, output_root, location):
    if location == 'wrist':
        support = source_root / 'PASOS' / 'PASOS_support_files'
    else:
        support = source_root / 'hip_data' / 'PASOS_hip' / 'support_files'
    randomization = read_randomization(support / 'PASOS_randomization.csv')
    train_validation = read_train_validation(support / 'train_val_split.csv')
    processed_root = output_root / 'processed' / location
    if not processed_root.is_dir():
        raise FileNotFoundError('Processed location directory is missing')
    source_counts = defaultdict(lambda: defaultdict(int))
    subjects_in_source = {'train': set(), 'test': set()}

    for subject_dir in processed_root.iterdir():
        if not subject_dir.is_dir():
            continue
        subject = subject_dir.name
        if subject not in randomization:
            raise ValueError('Processed subject is absent from randomization')
        source_split = randomization[subject]
        if source_split == 'train':
            if subject not in train_validation:
                raise ValueError('Processed development subject lacks train/validation assignment')
            partition = train_validation[subject]
            if partition == 'test':
                raise ValueError('Development subject has test assignment')
        else:
            if subject in train_validation and train_validation[subject] != 'test':
                raise ValueError('Test subject has development assignment')
            partition = 'test'
        subjects_in_source[source_split].add(subject)
        daily_files = sorted(subject_dir.glob('*.h5'))
        for target in ('development', partition) if source_split == 'train' else ('test',):
            source_counts[target]['participants'] += 1
            source_counts[target]['files'] += len(daily_files)
        for path in daily_files:
            with h5py.File(path, 'r') as source:
                labels = source['label'][:]
                sleeping = source['sleeping'][:]
                non_wear = source['non_wear'][:]
            if not (len(labels) == len(sleeping) == len(non_wear)):
                raise ValueError('H5 flag arrays have unequal lengths')
            eligible = (sleeping == 0) & (non_wear == 0) & np.isin(labels, (0, 1))
            values = {
                'all_10s_windows': len(labels),
                'usable_10s_windows': int(eligible.sum()),
                'sitting_windows': int((eligible & (labels == 0)).sum()),
                'non_sitting_windows': int((eligible & (labels == 1)).sum()),
            }
            for target in ('development', partition) if source_split == 'train' else ('test',):
                for key, value in values.items():
                    source_counts[target][key] += value

    if any(subjects_in_source[split] != {subject for subject, assigned in randomization.items()
                                               if assigned == split}
           for split in ('train', 'test')):
        raise ValueError('Processed subjects do not cover the randomization list')

    rows = []
    for partition in ('development', 'train', 'validation', 'test'):
        counts = source_counts[partition]
        all_windows = counts['all_10s_windows']
        row = {'location': location, 'partition': partition, **counts,
               'usable_percent': round(100 * counts['usable_10s_windows'] / all_windows, 2)
               if all_windows else 0.0}
        rows.append(row)
    all_processed = subjects_in_source['train'] | subjects_in_source['test']
    return rows, len(set(train_validation) - all_processed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path,
                        default=Path('/niddk-data-central/SOL'))
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for location in ('hip', 'wrist'):
        location_rows, extra_split_ids = summarize_location(
            args.source_root, args.output_root, location)
        rows.extend(location_rows)
        print('SPLIT_LIST_IDS_WITHOUT_RAW', location, extra_split_ids)
    fields = ('location', 'partition', 'participants', 'files', 'all_10s_windows',
              'usable_10s_windows', 'usable_percent', 'sitting_windows',
              'non_sitting_windows')
    report_dir = args.output_root / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / 'sol_pasos_data_summary.csv'
    with report.open('x', encoding='utf-8', newline='') as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print('SUMMARY', ','.join(str(row.get(field, 0)) for field in fields), flush=True)
    print('REPORT_SAVED', report, flush=True)


if __name__ == '__main__':
    main()
