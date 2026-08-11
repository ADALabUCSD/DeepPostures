---
layout: default
title: 2. Generating Predictions
parent: Getting Started
grand_parent: CHAP2.0
nav_order: 2
---

# Generating Predictions from CHAP2.0 Checkpoints
{: .no_toc }

You can use the submitted CHAP2.0 checkpoints to generate posture predictions
from pre-processed daily HDF5 files. This prediction-only workflow reads the
daily HDF5 directory directly and does **not** require
`create_dataset_split.py`.

To generate predictions, invoke `make_predictions.py` as follows:

```bash
python -m make_predictions \
  --data_dir <pre_processed_dir> \
  --checkpoint SUBMIT_RESULT/SOL_W/CHAP_FT/checkpoint-submit.pth \
  --prediction_dir <prediction_output_dir> \
  --batch_size 64 \
  --device cuda
```

Complete usage details of this script are as follows:

```text
usage: make_predictions [-h] --data_dir DATA_DIR --checkpoint CHECKPOINT
                        --prediction_dir PREDICTION_DIR
                        [--subjects SUBJECTS] [--model MODEL]
                        [--window_size WINDOW_SIZE]
                        [--batch_size BATCH_SIZE]
                        [--device {cpu,cuda}] [--seed SEED]

Run a CHAP wrist/hip checkpoint on CHAP-preprocessed data.

required arguments:
  --data_dir DATA_DIR
                        Pre-processed daily HDF5 directory:
                        <data_dir>/<subject_id>/<YYYY-MM-DD>.h5
  --checkpoint CHECKPOINT
                        Path to a CHAP .pth checkpoint
  --prediction_dir PREDICTION_DIR
                        Output directory for per-subject CSV files

optional arguments:
  -h, --help            show this help message and exit
  --subjects SUBJECTS   Comma-separated subject IDs. The default is to run on
                        all subject directories under --data_dir.
  --model MODEL         Model architecture (default: CHAP). The submitted
                        CHAP2.0 checkpoints use CHAP.
  --window_size WINDOW_SIZE
                        Number of 10-second windows per inference chunk
                        (default: 42)
  --batch_size BATCH_SIZE
                        Inference batch size (default: 64)
  --device {cpu,cuda}   Inference device (default: cuda)
  --seed SEED           Random seed (default: 0)
```

### Input format

The input directory should have the daily HDF5 layout produced by CHAP
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

Each daily HDF5 file should contain `data`, `label`, `time`, `non_wear`, and
`sleeping` fields. The script groups contiguous awake/worn 10-second windows
into chunks of `--window_size` windows. The default `--window_size 42`
corresponds to about 7 minutes. A trailing segment shorter than
`--window_size` is dropped.

### Output format

One CSV file is written per subject:

```text
<prediction_output_dir>/<subject_id>.csv
```

The output columns are:

```text
segment,timestamp,prediction[,label]
```

The `label` column is included only if the input contains any labeled windows.
If the input labels are all `-1`, prediction-only output is written.

### Tail windows

CHAP2.0 prediction drops trailing windows that do not fill a complete
`--window_size` chunk. Unlike the MSSE-2021 prediction script, this script does
not currently provide `zero` or `wrap` padding options.

### Checkpoints

Pass one of the submitted CHAP2.0 checkpoints under `CHAP2/SUBMIT_RESULT/`, or
a CHAP checkpoint trained with `main_finetune.py`.

CHAP-ZS means zero-shot prediction: the CHAP/MSSE-2021 pre-trained checkpoint
is applied directly to the target dataset without fine-tuning. The separate
CHAP-ZS folders correspond to different target datasets or sensor locations,
not to separately trained zero-shot models.

CHAP-FT checkpoints are fine-tuned from CHAP/MSSE-2021 weights on the target
dataset. See [Datasets and Models]({{ site.baseurl }}{% link chap2/datasets_and_models.md %})
for more details on the submitted checkpoints and recommended defaults.

If you want to fine-tune a checkpoint on your own labeled dataset, see
[Fine-Tuning Your Own Model]({{ site.baseurl }}{% link chap2/finetuning.md %}).
