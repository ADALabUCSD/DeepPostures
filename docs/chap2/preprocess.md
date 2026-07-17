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
                        Path to the valid days file
  --sleep-logs-file SLEEP_LOGS_FILE
                        Path to the sleep logs file
  --wear-logs-file WEAR_LOGS_FILE
                        Path to the wear logs file
  --loc {hip,wrist}     Wear location to use when parsing iWatch non-wear files
  --non-wear-times-file NON_WEAR_TIMES_FILE
                        Path to non-wear times file
  --activpal-dir ACTIVPAL_DIR
                        ActivPAL data directory
  --event-file          Interpret ActivPAL CSVs as event-format files with
                        Time and Interval (s) columns. Leave unset for
                        1-second epoch ActivPAL files.
  --n-start-id N_START_ID
                        The index of the starting character of the ID in GT3X
                        file names. Indexing starts with 1.
  --n-end-id N_END_ID   The index of the ending character of the ID in GT3X
                        file names
  --expression-after-id EXPRESSION_AFTER_ID
                        String or list of strings used to identify the ID from
                        the GT3X file name. The first split will be used as the
                        file name.
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
  --silent              Whether to hide info messages
  --mp MP               Number of concurrent workers
  --gzipped             Whether the raw data is gzipped. The extension should
                        be .csv.gz.
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
| `--sleep-logs-file` | CHAP1.0 format: `ID, Date.In.Bed, Time.In.Bed, Date.Out.Bed, Time.Out.Bed`; CHAP2/SOL-style format: `id, startsleep, endsleep`. |
| `--wear-logs-file` | `shortid, startwear, endwear`; used as a complement to sleep logs when wear intervals are available. |
| `--non-wear-times-file` | CHAP1.0 format: `ID, Date.nw.start, Time.nw.start, Date.nw.end, Time.nw.end`; iWatch-style format includes `id, wearloc, nw_dt, int.min, weardate, ..., loc`; SOL-style format: `id, startNW, endNW`. |
| `--activpal-dir` | Directory containing ActivPAL label files. Use `--event-file` for event-format files with `Time`, `Interval (s)`, and `ActivityCode` columns. Leave `--event-file` unset for 1-second epoch files using `TS_LOCAL_COR` and `PL_ACTIVITY_NEW...` columns. |

For iWatch non-wear files that include both hip and wrist records, pass
`--loc hip` or `--loc wrist` to select the device location to process.

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
