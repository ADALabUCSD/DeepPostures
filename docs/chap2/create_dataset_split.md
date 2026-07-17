---
layout: default
title: 0. Creating Dataset Splits
parent: Advanced Usages
grand_parent: CHAP2.0
nav_order: 0
---

# Creating Dataset Splits
{: .no_toc }

CHAP2.0 fine-tuning uses split HDF5 files rather than reading the daily HDF5
files directly. After preprocessing, run `create_dataset_split.py` to group
10-second windows into fixed-length samples and write train, validation, and
test files.

Prediction with `make_predictions.py` does **not** require this step. This
step is needed for fine-tuning and split-based evaluation with
`main_finetune.py`.

Invoke the script as follows:

```bash
python create_dataset_split.py \
  --data_dir <pre_processed_dir> \
  --split_csv <subject_split_csv> \
  --output_dir <split_data_dir>
```

Complete usage details of this script are as follows:

```text
usage: create_dataset_split.py [-h] [--demo] [--data_dir DATA_DIR]
                               [--split_csv SPLIT_CSV]
                               [--output_dir OUTPUT_DIR]
                               [--window_size WINDOW_SIZE]
                               [--flush_threshold FLUSH_THRESHOLD]

required arguments for a full run:
  --data_dir DATA_DIR
                        Pre-processed daily HDF5 directory
  --split_csv SPLIT_CSV
                        CSV with subject_id and split columns
  --output_dir OUTPUT_DIR
                        Output directory for split HDF5 files

optional arguments:
  -h, --help            show this help message and exit
  --demo                Run demo with a single subject from
                        DEMO/CHAP1_preprocess_demo/
  --window_size WINDOW_SIZE
                        Number of 10-second windows per sample (default: 42)
  --flush_threshold FLUSH_THRESHOLD
                        Number of samples buffered before writing to HDF5
                        (default: 1000)
```

### Input: daily HDF5 directory

The `--data_dir` argument should point to the daily HDF5 output from
preprocessing:

```text
pre_processed_dir/
├── subject_001/
│   ├── 2023-01-01.h5
│   ├── 2023-01-02.h5
│   └── ...
└── subject_002/
    └── ...
```

Each daily file contains 10-second windows with fields such as `data`, `label`,
`time`, `non_wear`, and `sleeping`.

### Input: subject split CSV

The `--split_csv` file should assign each subject to a split:

```csv
subject_id,split
subject_001,train
subject_002,validation
subject_003,test
```

The `subject_id` values must match the folder names under `--data_dir`.

### Output: split HDF5 files

The output directory should contain:

```text
split_data_dir/
├── 10s_train.h5
├── 10s_val.h5
└── 10s_test_complete.h5
```

Each file contains:

| Field | Shape | Description |
|-------|-------|-------------|
| `x` | `(N, 42, 100, 3)` | Accelerometer data grouped into 42 consecutive 10-second windows. |
| `y` | `(N, 42)` | Labels for each 10-second window in the sample. |
| `timestamp` | `(N, 42)` | Unix timestamps for the windows. |
| `subject_id` | `(N,)` | Subject identifier for each sample. |
| `std` | `(N, 42)` | Mean standard deviation of the accelerometer signal for each 10-second window. |

If you change `--window_size`, the second dimension of `x`, `y`, `timestamp`,
and `std` changes accordingly.

### Filtering behavior

During split creation, the iterator skips sleeping windows and non-wear
windows. For training-style iteration, unlabeled windows (`label == -1`) are
also skipped. A trailing sequence shorter than `--window_size` is dropped
because CHAP expects fixed-length samples.

### Demo

To run the bundled split creation demo:

```bash
python create_dataset_split.py --demo
```

This reads:

```text
DEMO/CHAP1_preprocess_demo/demo_subject/2018-06-25.h5
```

and writes:

```text
DEMO/demo_output/10s_train.h5
DEMO/demo_output/10s_val.h5
DEMO/demo_output/10s_test_complete.h5
```
