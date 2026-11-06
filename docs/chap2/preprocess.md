---
layout: default
title: 1. Data Preprocessing
parent: Getting Started
grand_parent: CHAP2.0
nav_order: 1
---

# CHAP2.0 Data Preprocessing
{: .no_toc }

First, you need to create pre-processed daily HDF5 files from the source
accelerometer data. To do this, invoke the CHAP2.0 preprocessing script as
follows:

```bash
python pre_process/pre_process_data.py --gt3x-dir <gt3x_data_dir> --pre-processed-dir <output_dir>
```

CHAP2.0 preprocessing supports CHAP1.0-style files as well as additional
iWatch and SOL/PASOS wear, non-wear, sleep-log, and ActivPAL label formats.

Complete usage details of this script are as follows:

The two directory arguments are required and have no default. Optional file
arguments default to `None` (unset), and Boolean flags default to `False`.

```text
usage: pre_process_data.py [-h] --gt3x-dir GT3X_DIR --pre-processed-dir PRE_PROCESSED_DIR
                           [--valid-days-file VALID_DAYS_FILE]
                           [--sleep-logs-file SLEEP_LOGS_FILE]
                           [--wear-logs-file WEAR_LOGS_FILE]
                           [--loc {hip,wrist}]
                           [--non-wear-times-file NON_WEAR_TIMES_FILE]
                           [--activpal-dir ACTIVPAL_DIR] [--event-file]
                           [--n-start-id N_START_ID] [--n-end-id N_END_ID]
                           [--expression-after-id EXPRESSION_AFTER_ID]
                           [--window-size WINDOW_SIZE]
                           [--gt3x-frequency GT3X_FREQUENCY]
                           [--down-sample-frequency DOWN_SAMPLE_FREQUENCY]
                           [--activpal-label-map ACTIVPAL_LABEL_MAP]
                           [--silent] [--mp MP] [--gzipped]

Argument parser for preprocessing the input data.

required arguments:
  --gt3x-dir GT3X_DIR
                        GT3X data directory
  --pre-processed-dir PRE_PROCESSED_DIR
                        Pre-processed data directory

optional arguments:
  -h, --help            show this help message and exit
  --valid-days-file VALID_DAYS_FILE
                        Path to the valid days file (default: None)
  --sleep-logs-file SLEEP_LOGS_FILE
                        Path to the sleep logs file (default: None)
  --wear-logs-file WEAR_LOGS_FILE
                        Path to the wear logs file (default: None)
  --loc {hip,wrist}     Location for iWatch-style non-wear files; required for
                        that format (default: None; see explanation below)
  --non-wear-times-file NON_WEAR_TIMES_FILE
                        Path to non-wear times file (default: None)
  --activpal-dir ACTIVPAL_DIR
                        ActivPAL data directory (default: None)
  --event-file          Interpret ActivPAL CSVs as event-format files with
                        Time and Interval (s) columns. Leave unset for
                        1-second epoch ActivPAL files (default: False).
  --n-start-id N_START_ID
                        The index of the starting character of the ID in GT3X
                        file names. Indexing starts with 1; requires
                        --n-end-id (default: None).
  --n-end-id N_END_ID   The index of the ending character of the ID in GT3X
                        file names (default: None)
  --expression-after-id EXPRESSION_AFTER_ID
                        Delimiter string used to extract the ID from the GT3X
                        file name; use the prefix before its first occurrence
                        (default: None).
  --window-size WINDOW_SIZE
                        Window size in seconds on which predictions are made
                        (default: 10)
  --gt3x-frequency GT3X_FREQUENCY
                        GT3X device frequency in Hz (default: 30)
  --down-sample-frequency DOWN_SAMPLE_FREQUENCY
                        Downsample frequency in Hz for GT3X data (default: 10)
  --activpal-label-map ACTIVPAL_LABEL_MAP
                        ActivPAL label vocabulary (default:
                        {"0": 0, "1": 1, "2": 1})
  --silent              Whether to hide info messages (default: False)
  --mp MP               Number of concurrent workers; None runs serially
                        (default: None)
  --gzipped             Whether the raw data is gzipped. The extension should
                        be .csv.gz (default: False).
```

### Input and annotation files

CHAP2.0 accepts the same GT3X CSV, valid-day, sleep-log, non-wear, and
ActivPAL inputs described in the
[MSSE-2021 preprocessing guide]({{ site.baseurl }}{% link getting_started/preprocess.md %}).
It also supports additional iWatch and SOL/PASOS-style annotation files parsed
by header names.

| File | Supported formats and notes |
|------|-----------------------------|
| `--gt3x-dir` | Directory containing raw accelerometer CSV files. Use `--gt3x-frequency` to specify the raw sampling frequency and `--down-sample-frequency` to set the output frequency. |
| `--valid-days-file` | CHAP1.0-style valid-day CSV with subject/date rows. |
| `--sleep-logs-file` | CHAP1.0 format: `ID, Date.In.Bed, Time.In.Bed, Date.Out.Bed, Time.Out.Bed`; three-column SOL-style formats: `ID,startsleep,endsleep` or `ID,startSL,endSL`, accepting `M/D/YY HH:MM` or `YYYY-MM-DD HH:MM:SS`. |
| `--wear-logs-file` | `shortid, startwear, endwear`; provides awake intervals for subjects without sleep-log entries. |
| `--non-wear-times-file` | CHAP1.0 format: `ID, Date.nw.start, Time.nw.start, Date.nw.end, Time.nw.end`; iWatch-style format includes `id, wearloc, nw_dt, int.min, weardate, ..., loc`; SOL-style format: `ID,startNW,endNW`, accepting `M/D/YY HH:MM` or `YYYY-MM-DD HH:MM:SS`. |
| `--activpal-dir` | Directory containing ActivPAL label files. Use `--event-file` for event-format files with `Time`, `Interval (s)`, and `ActivityCode` columns. Leave `--event-file` unset for 1-second epoch files using `TS_LOCAL` and `PL_ACTIVITY_NEW...` columns. `TS_LOCAL_COR` is not used for ActiGraph alignment. |

`--loc` is optional overall and defaults to `None`, selecting neither hip nor
wrist automatically. For iWatch-style non-wear files, explicitly pass
`--loc hip` or `--loc wrist`, even for a single-location file. It filters the
`loc` column, not the values in `wearLoc`. Other studies can use the same
supported file structure; a `wearLoc` column alone is insufficient.
For supported formats without location fields, such as SOL, or when no
non-wear file is supplied, `--loc` has no effect and produces a warning unless
`--silent` is set. Unsupported formats raise an error.

For datasets with mixed sampling frequencies, split the raw files by frequency
and run preprocessing separately with the matching `--gt3x-frequency`. For
example, if a SOL/PASOS input directory contains both 60 Hz and 80 Hz files,
process those subject groups separately rather than relying on one mixed run.

### Output format

The output directory contains one folder per subject and one HDF5 file per day:

```text
pre_processed_dir/
├── subject_001/
│   ├── 2023-01-01.h5
│   ├── 2023-01-02.h5
│   └── ...
└── subject_002/
    └── ...
```

Each daily `.h5` file contains 10-second windows:

| Field | Shape | Description |
|-------|-------|-------------|
| `data` | `(N, 100, 3)` | Accelerometer readings at 10 Hz. |
| `label` | `(N,)` | `0` = sitting, `1` = not sitting, `-1` = unknown. |
| `time` | `(N,)` | Unix timestamp for each 10-second window. |
| `non_wear` | `(N,)` | `1` = device not worn. |
| `sleeping` | `(N,)` | `1` = asleep. |

The daily HDF5 files can be used directly for prediction with
`make_predictions.py`. Fine-tuning requires an additional split creation step;
see [Creating Dataset Splits]({{ site.baseurl }}{% link chap2/create_dataset_split.md %}).

### Speed up

CHAP2.0 preprocessing supports the same `--mp` and `--gzipped` options as
MSSE-2021. If preprocessing is slow, see the
[parallel processing guide]({{ site.baseurl }}{% link advanced/parallel_processing.md %}).
If the raw CSV files are large, see the
[data compression guide]({{ site.baseurl }}{% link advanced/compression.md %}).
