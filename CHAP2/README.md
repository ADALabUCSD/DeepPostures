# CHAP2.0

CHAP2.0 contains PyTorch code for applying and fine-tuning CHAP models on
wrist and hip accelerometer posture classification tasks, including iWatch and
SOL/PASOS workflows.

The main supported workflows are:

- **CHAP-ZS**: zero-shot prediction using CHAP checkpoints.
- **CHAP-FT**: fine-tuning CHAP checkpoints on a target wrist or hip dataset.

# Table of Contents

- [Data Workflow](#data-workflow)
- [Repository Structure](#repository-structure)
- [Pre-Processing Data](#pre-processing-data)
- [Creating Train/Validation/Test Splits](#creating-trainvalidationtest-splits)
- [Generating Predictions from CHAP Checkpoints](#generating-predictions-from-chap-checkpoints)
- [Fine-Tuning CHAP](#fine-tuning-chap)
- [Submitted Checkpoints](#submitted-checkpoints)
- [Demo](#demo)
- [Notes](#notes)

## Data Workflow

CHAP2.0 uses the same daily HDF5 format produced by CHAP preprocessing:

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

For training and evaluation, `create_dataset_split.py` groups these 10-second
windows into longer samples:

```text
split_data_dir/
├── 10s_train.h5
├── 10s_val.h5
└── 10s_test_complete.h5
```

The split files contain `x` with shape `(N, 42, 100, 3)` and `y` with shape
`(N, 42)`.

## Repository Structure

```text
CHAP2/
├── pre_process/                 # Preprocessing and label-distribution utilities
├── DEMO/                        # Small demo input/output files
├── SUBMIT_RESULT/               # Submitted CHAP checkpoints
├── script/                      # Example training and prediction scripts
├── util/                        # Dataset, loss, metric, and training utilities
├── create_dataset_split.py      # Daily HDF5 -> split HDF5 conversion
├── main_finetune.py             # CHAP fine-tuning and split-based evaluation
├── make_predictions.py          # Prediction-only script for CHAP checkpoints
├── chap_model.py                # CHAP and CHAP attention model definitions
├── engine_finetune.py           # Train/evaluate loops
├── example_models.py            # Model loading and inference examples
└── requirements.txt             # Additional Python dependencies
```

## Pre-Processing Data

Use `pre_process/pre_process_data.py` to convert raw accelerometer and optional
annotation files into daily HDF5 files:

```bash
python pre_process/pre_process_data.py \
  --gt3x-dir <raw_csv_dir> \
  --pre-processed-dir <output_dir>
```

CHAP2.0 preprocessing supports CHAP1.0-style files as well as iWatch and
SOL/PASOS-specific wear, non-wear, sleep-log, and ActivPAL label formats.
Common optional arguments include:

```text
--valid-days-file
--sleep-logs-file
--wear-logs-file
--non-wear-times-file
--activpal-dir
--event-file
--loc {hip,wrist}
--window-size
--gt3x-frequency
--down-sample-frequency
```

Argument summary:

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--gt3x-dir` | yes | - | Directory containing raw accelerometer CSV files. |
| `--pre-processed-dir` | yes | - | Output directory for daily HDF5 files. |
| `--valid-days-file` | no | `None` | Optional valid-day CSV. |
| `--sleep-logs-file` | no | `None` | Optional sleep-log CSV. |
| `--wear-logs-file` | no | `None` | Optional wear-log CSV. |
| `--non-wear-times-file` | no | `None` | Optional non-wear interval CSV. |
| `--activpal-dir` | no | `None` | Directory containing ActivPAL label files. |
| `--event-file` | no | `False` | Treat ActivPAL files as event-format files. |
| `--loc` | no | `None` | iWatch location, either `hip` or `wrist`. |
| `--n-start-id` | no | `None` | Starting character index for extracting subject IDs from filenames. |
| `--n-end-id` | no | `None` | Ending character index for extracting subject IDs from filenames. |
| `--expression-after-id` | no | `None` | Split expression used to extract subject IDs from filenames. |
| `--window-size` | no | `10` | Window size in seconds for daily HDF5 windows. |
| `--gt3x-frequency` | no | `30` | Raw accelerometer sampling frequency in Hz. |
| `--down-sample-frequency` | no | `10` | Output sampling frequency in Hz. |
| `--activpal-label-map` | no | `{"0": 0, "1": 1, "2": 1}` | Mapping from ActivPAL labels to CHAP binary labels. |
| `--silent` | no | `False` | Hide informational messages. |
| `--mp` | no | `None` | Number of multiprocessing workers. |
| `--gzipped` | no | `False` | Read gzipped raw CSV files. |

For CHAP fine-tuning and labeled evaluation, unlabeled windows (`label == -1`)
should be excluded. `create_dataset_split.py` drops these windows for training
splits; `make_predictions.py` can still generate prediction-only outputs when
labels are unavailable.

## Creating Train/Validation/Test Splits

For fine-tuning, first create split HDF5 files from daily HDF5 files:

```bash
python create_dataset_split.py \
  --data_dir <pre_processed_dir> \
  --split_csv <subject_split_csv> \
  --output_dir <split_data_dir>
```

The split CSV should assign each subject to a split:

```csv
subject_id,split
subject_001,train
subject_002,validation
subject_003,test
```

The output directory should contain:

```text
10s_train.h5
10s_val.h5
10s_test_complete.h5
```

Argument summary:

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data_dir` | yes | - | Directory of daily HDF5 files. |
| `--split_csv` | yes | - | CSV with `subject_id` and `split` columns. |
| `--output_dir` | yes | - | Output directory for split HDF5 files. |
| `--window_size` | no | `42` | Number of 10-second windows per sample. |
| `--flush_threshold` | no | `1000` | Number of samples buffered before writing to HDF5. |
| `--demo` | no | `False` | Run the bundled demo split creation. |

## Generating Predictions from CHAP Checkpoints

For prediction-only use cases, `make_predictions.py` reads daily HDF5 files
directly and does not require train/validation/test split files:

```bash
python -m make_predictions \
  --data_dir <pre_processed_dir> \
  --checkpoint SUBMIT_RESULT/SOL_W/CHAP_FT/checkpoint-submit.pth \
  --prediction_dir <prediction_output_dir> \
  --batch_size 64 \
  --device cuda
```

One CSV file is written per subject. The output columns are:

```text
segment,timestamp,prediction[,label]
```

`make_predictions.py` currently supports CHAP checkpoints.

Argument summary:

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data_dir` | yes | - | Daily HDF5 directory: `<data_dir>/<subject_id>/<YYYY-MM-DD>.h5`. |
| `--checkpoint` | yes | - | Path to a CHAP `.pth` checkpoint. |
| `--prediction_dir` | yes | - | Output directory for per-subject CSV files. |
| `--subjects` | no | `None` | Comma-separated subject IDs; default is all subject directories. |
| `--model` | no | `CHAP` | Model architecture. Current prediction script supports `CHAP`. |
| `--window_size` | no | `42` | Number of 10-second windows per inference chunk. |
| `--batch_size` | no | `64` | Inference batch size. |
| `--device` | no | `cuda` | Inference device: `cuda` or `cpu`. |
| `--seed` | no | `0` | Random seed. |

## Fine-Tuning CHAP

Fine-tuning uses the split HDF5 files created by `create_dataset_split.py`.
From inside `CHAP2/`:

```bash
torchrun --nproc_per_node=<num_gpus> -m main_finetune \
  --data_path <split_data_dir> \
  --model CHAP \
  --checkpoint ../MSSE-2021/pre-trained-models-pt/CHAP_ALL_ADULTS.pth \
  --output_dir <checkpoint_output_dir> \
  --remark <experiment_name> \
  --blr 1e-3 \
  --epochs 10 \
  --warmup_epochs 2 \
  --batch_size 16 \
  --weight_decay 1e-3 \
  --use_data_aug 1
```

Example scripts are provided in `script/`:

```text
script/chap_ft_sol.sh       # SOL/PASOS CHAP fine-tuning example
script/chap_ft_iwatch.sh    # iWatch CHAP fine-tuning example
script/chap_scratch_sol.sh  # SOL/PASOS training-from-scratch example
```

These scripts contain lab-specific paths and should be edited before use.

Argument summary:

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data_path` | yes | - | Directory containing `10s_train.h5`, `10s_val.h5`, and `10s_test_complete.h5`. |
| `--model` | no | `CHAP` | Model architecture. Official CHAP2.0 uses `CHAP`. |
| `--checkpoint` | no | `None` | Checkpoint used to initialize/fine-tune the model. |
| `--output_dir` | no | `./output` | Directory for training checkpoints and outputs. |
| `--log_dir` | no | `None` | Directory for W&B logs; unset disables W&B logging. |
| `--remark` | no | `Debug` | Experiment name. |
| `--epochs` | no | `20` | Number of training epochs. |
| `--batch_size` | no | `64` | Batch size per process/GPU. |
| `--accum_iter` | no | `1` | Gradient accumulation steps. |
| `--blr` | no | `5e-4` | Base learning rate. |
| `--lr` | no | `None` | Absolute learning rate; overrides `--blr` when set. |
| `--weight_decay` | no | `5e-2` | Weight decay for optimizer. |
| `--warmup_epochs` | no | `2` | Number of learning-rate warmup epochs. |
| `--pos_weight` | no | `1.0` | Positive-class weight for BCE loss. |
| `--use_focal_loss` | no | `False` | Use focal loss instead of BCEWithLogitsLoss. |
| `--subset_ratio` | no | `1.0` | Fraction of training data to use. |
| `--use_data_aug` | no | `1` | Enable data augmentation. |
| `--eval` | no | `None` | Run evaluation from this checkpoint instead of training. |
| `--make_prediction` | no | `False` | Save subject-level prediction CSVs during evaluation. |
| `--prediction_dir` | no | `None` | Output directory for prediction CSVs. |
| `--device` | no | `cuda` | Training/evaluation device. |
| `--num_workers` | no | `4` | DataLoader worker count. |
| `--pin_mem` / `--no_pin_mem` | no | `pin_mem=True` | Enable or disable pinned CPU memory. |

## Submitted Checkpoints

The submitted CHAP checkpoints are stored under `SUBMIT_RESULT/`:

Recommended defaults:

| Checkpoint | Recommended for |
|------------|-----------------|
| `SOL_W/CHAP_FT/checkpoint-submit.pth` | ActiGraph wrist data. |
| `iWatch_H/CHAP-ZS/checkpoint/checkpoint-submit.pth` | ActiGraph hip data trained on ACT and AusDiab. |

Additional submitted checkpoints are retained for reproducibility and
comparison:

| Checkpoint | Description |
|------------|-------------|
| `iWatch_W/CHAP-FT/checkpoint/checkpoint-submit.pth` | iWatch wrist CHAP-FT checkpoint. |
| `iWatch_W/CHAP-ZS/checkpoint/checkpoint-submit.pth` | iWatch wrist CHAP-ZS checkpoint. |
| `iWatch_H/CHAP-FT/checkpoint/checkpoint-submit.pth` | iWatch hip CHAP-FT checkpoint. |

## Demo

A small CPU demo is included:

```bash
cd CHAP2
bash script/make_prediction_cpu_demo.sh
```

The demo reads `DEMO/CHAP1_preprocess_demo/` and writes prediction CSV files to
`DEMO/demo_prediction/`.

## Notes

- CHAP2.0 is kept as a directory parallel to `MSSE-2021/`.
- The official CHAP2.0 public merge focuses on CHAP-ZS and CHAP-FT.
- Detailed tutorials and website documentation should be used for end-to-end
  examples and dataset-specific guidance.
