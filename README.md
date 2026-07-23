CHAP 2.0 — Wrist & Hip Accelerometer Posture Classification
============================================================

This repository takes **accelerometer data** (from an ActiGraph)
and labels every 10 seconds as **sitting** or **not sitting**. It ships every
model from the paper, each usable on both hip and wrist data (including
SOL/PASOS):

- **CHAP-ZS**: Zero-shot CHAP model (no finetuning)
- **CHAP-FT**: Finetuned CHAP model for posture classification
- **CHAP-ViT**: Vision Transformer variants (ViT-base, ViT-small, ViT-tiny)

### Which path are you on?

| I want to... | Go to |
|--------------|-------|
| **Get sitting/not-sitting predictions from my ActiGraph data** (most users) | [Predict on Your Own Data](#predict-on-your-own-data) |
| Just check the code runs on my machine | [Try the Demo](#try-the-demo-cpu-no-gpu) |
| Train / finetune a model on my own labeled data | [Finetune Your Own Model](#finetune-your-own-model) |
| Look up file formats, repo layout, or checkpoint details | [Reference](#reference) |

> For the legacy TensorFlow code and earlier publications, see the `master` branch.


Installation
------------

- Python 3.8+
- [Conda](https://github.com/conda-forge/miniforge/releases/latest) (Miniforge recommended)
- A GPU is strongly recommended for training, but **not needed for prediction**.

```bash
conda create -n chap python=3.10 -y
conda activate chap

# From the repo root:
pip install -r CHAP2/requirements.txt
```


Try the Demo (CPU, no GPU)
--------------------------

Before using your own data, run the bundled toy example to confirm your setup
works. It predicts on one subject of wrist data — **no GPU required**.

```bash
cd CHAP2
bash script/make_prediction_cpu_demo.sh
```

This loads the SOL/PASOS wrist checkpoint and runs prediction on the bundled,
already-preprocessed data:

| Input | Output |
|-------|--------|
| `DEMO/CHAP1_preprocess_demo/<subject_id>/<YYYY-MM-DD>.h5` + `SUBMIT_RESULT/SOL_W/CHAP_FT/checkpoint-submit.pth` | `DEMO/demo_prediction/<subject_id>.csv` |

If a CSV appears in `DEMO/demo_prediction/`, you're ready to run on your own data.


Predict on Your Own Data
========================

The whole flow is two steps:

```
raw ActiGraph GT3X CSV  ──Step 1──►  daily HDF5 files  ──Step 2──►  prediction CSV
   (one CSV per subject)              (per subject)                 (one per subject)
```

You do **not** need to train anything, and you do **not** need labeled data or
train/val/test splits — those are only for [finetuning](#finetune-your-own-model).


Step 1 — Preprocess: raw ActiGraph CSV → HDF5
---------------------------------------------

**Using an ActiGraph GT3X?** Run
[`MSSE_2021_pt/pre_process_data.py`](MSSE_2021_pt/pre_process_data.py).

Your input should be one **raw 30 Hz ActiGraph GT3X CSV per subject**, all in one
directory — a 10-line ActiLife header followed by `Accelerometer X,Y,Z` rows (see
the sample at
[`MSSE_2021_pt/example/example_30Hz.csv`](MSSE_2021_pt/example/example_30Hz.csv)).

```bash
python MSSE_2021_pt/pre_process_data.py \
  --gt3x-dir              /path/to/raw_gt3x_csv_dir \
  --pre-processed-dir     /path/to/save_preprocessed \
  --gt3x-frequency        30 \
  --down-sample-frequency 10 \
  --window-size           10 \
  --mp                    4
```

Only `--gt3x-dir` and `--pre-processed-dir` are required. The label / sleep /
wear / non-wear / valid-day files are **optional** — you need them only if you
plan to finetune with ground-truth labels (otherwise the label is written as
`-1` = unknown).

<details>
<summary>Complete <code>pre_process_data.py</code> options</summary>

    required:
      --gt3x-dir DIR                GT3X 30Hz CSV directory (one CSV per subject)
      --pre-processed-dir DIR       Output directory for daily HDF5

    optional:
      --activpal-dir DIR            ActivPAL events CSVs = ground-truth labels (training only)
      --valid-days-file CSV         Valid-day file (ID, Date.valid.day)
      --sleep-logs-file CSV         Sleep intervals
      --wear-logs-file CSV          Wear intervals (complement of sleep)
      --non-wear-times-file CSV     Non-wear intervals
      --loc {hip,wrist}             Sensor location filter for non-wear file
      --n-start-id / --n-end-id     Char indices to slice subject ID from filename (1-based)
      --expression-after-id STR     Split filename on this string to get subject ID
      --window-size N               Window length in seconds (default: 10)
      --gt3x-frequency N            Device sample rate in Hz (default: 30)
      --down-sample-frequency N     Target rate in Hz (default: 10)
      --activpal-label-map JSON     ActivPAL code → label map (default: {"0":0,"1":1,"2":1})
      --mp N                        Parallel workers, set to #cores (default: none)
      --gzipped                     Raw files are .csv.gz
      --silent                      Suppress info logs

> **Note:** the `--gt3x-frequency 60` and `80` branches are SOL/PASOS-specific
> and hard-code an internal support-file path — for your own data use the
> default 30 Hz path.

</details>

> **Not using an ActiGraph?** This script only understands the ActiLife GT3X CSV
> export. For any other device, convert your data into the daily-HDF5 format
> documented in
> [Data Formats](#source-data-output-of-chap1-preprocessing), then continue to
> Step 2.


Step 2 — Predict: run a trained model
-------------------------------------

Point [`make_predictions.py`](CHAP2/make_predictions.py) at the folder you
produced in Step 1. Run it from inside `CHAP2/`:

```bash
cd CHAP2
python -m make_predictions \
  --data_dir       /path/to/save_preprocessed \
  --checkpoint     SUBMIT_RESULT/SOL_W/CHAP_FT/checkpoint-submit.pth \
  --prediction_dir /path/to/save_predictions \
  --batch_size 64 \
  --device cuda        # use "cpu" if you have no GPU
```

**Which `--checkpoint`?** Pick the one that matches your device and wear
location:

**Recommended** — start here:

| Checkpoint | Notes |
|-----------|-------|
| `SUBMIT_RESULT/SOL_W/CHAP_FT/checkpoint-submit.pth` | wrist — SOL/PASOS-tuned |
| `SUBMIT_RESULT/iWatch_H/CHAP-ZS/checkpoint/checkpoint-submit.pth` | hip — ACT and AusDiab trained |

Other alternatives:

| Checkpoint | Notes |
|-----------|-------|
| `SUBMIT_RESULT/iWatch_W/CHAP-FT/checkpoint/checkpoint-submit.pth` | wrist — iWatch-finetuned |
| `SUBMIT_RESULT/iWatch_H/CHAP-FT/checkpoint/checkpoint-submit.pth` | hip — iWatch-finetuned |
| `SUBMIT_RESULT/iWatch_W/CHAP-ZS/checkpoint/checkpoint-submit.pth` | wrist — zero-shot (no finetuning) baseline |

**Output.** One CSV per subject lands in `--prediction_dir`, with columns
`segment, timestamp, prediction` (where `prediction` is `0` = sitting,
`1` = not sitting). A `label` column is added only if your input had labels.

That's it — for prediction you never touch training scripts or splits. If you
*do* have labels and want accuracy / F1 / a confusion matrix, use the
[evaluation flow](#finetune-your-own-model) (Step T3) instead.

<details>
<summary>Complete <code>make_predictions.py</code> options</summary>

    usage: make_predictions [-h] --data_dir DATA_DIR --checkpoint CHECKPOINT
                            --prediction_dir PREDICTION_DIR
                            [--subjects SUBJECTS] [--model MODEL]
                            [--window_size WINDOW_SIZE]
                            [--batch_size BATCH_SIZE]
                            [--device {cpu,cuda}] [--seed SEED]

    Run a CHAP wrist/hip checkpoint on CHAP1-preprocessed data.

    required arguments:
      --data_dir DATA_DIR   CHAP1-preprocessed dir:
                            <data_dir>/<subject_id>/<YYYY-MM-DD>.h5
      --checkpoint CHECKPOINT
                            Path to model .pth (see "Available Checkpoints" below)
      --prediction_dir PREDICTION_DIR
                            Output dir for per-subject CSVs

    optional arguments:
      --subjects SUBJECTS   Comma-separated subject IDs to predict on
                            (default: all subdirs of --data_dir)
      --model MODEL         Model architecture (default: CHAP).
                            All bundled SUBMIT_RESULT checkpoints are CHAP.
      --window_size WINDOW_SIZE
                            10s windows per inference chunk; must match training
                            (default: 42)
      --batch_size BATCH_SIZE
                            Inference batch size (default: 64). Reduce if OOM.
      --device {cpu,cuda}   Inference device (default: cuda)
      --seed SEED           Random seed (default: 0)

</details>


Finetune Your Own Model
=======================

Only needed if you want to **train** a model on your own labeled data instead of
using the provided checkpoints. Training needs pre-built train/val/test HDF5
splits, so there's an extra prep step.

First run [Step 1](#step-1--preprocess-raw-actigraph-csv--hdf5) above to get
daily HDF5 files (with labels — supply `--activpal-dir`), then:

**Step T1 — Build splits.** Prepare a split CSV (see
[Train/val/test split file](#trainvaltest-split-file)), then run:

```bash
cd CHAP2
python create_dataset_split.py \
  --data_dir /path/to/save_preprocessed \
  --split_csv /path/to/your_split.csv \
  --output_dir /path/to/your_split_data
```

**Step T2 — Train.**

```bash
cd CHAP2
torchrun --nproc_per_node=<num_gpus> -m main_finetune \
  --data_path /path/to/your_split_data \
  --model CHAP \
  --checkpoint MSSE_2021_pt/pre-trained-models-pt/CHAP_ALL_ADULTS.pth \
  --output_dir /path/to/save_checkpoints \
  --remark my_experiment \
  --blr 1e-3 \
  --epochs 10 \
  --warmup_epochs 2 \
  --batch_size 16 \
  --weight_decay 1e-3 \
  --use_data_aug 1
```

For single-GPU or CPU training, use `python -m main_finetune` instead of `torchrun`.

<details>
<summary>Complete <code>main_finetune</code> training options</summary>

    usage: main_finetune [-h] [--config CONFIG] [--data_path DATA_PATH]
                         [--model MODEL] [--checkpoint CHECKPOINT]
                         [--output_dir OUTPUT_DIR] [--log_dir LOG_DIR]
                         [--remark REMARK] [--epochs EPOCHS]
                         [--batch_size BATCH_SIZE] [--accum_iter ACCUM_ITER]
                         [--blr BLR] [--lr LR] [--min_lr MIN_LR]
                         [--layer_decay LAYER_DECAY]
                         [--warmup_epochs WARMUP_EPOCHS]
                         [--weight_decay WEIGHT_DECAY]
                         [--clip_grad CLIP_GRAD]
                         [--use_data_aug USE_DATA_AUG]
                         [--subset_ratio SUBSET_RATIO]
                         [--use_focal_loss] [--pos_weight POS_WEIGHT]
                         [--drop_path_rate DROP_PATH_RATE]
                         [--device DEVICE] [--seed SEED]
                         [--resume RESUME] [--start_epoch START_EPOCH]
                         [--num_workers NUM_WORKERS]
                         [--nb_classes NB_CLASSES]
                         [--dist_eval] [--pin_mem | --no_pin_mem]

    Finetune a model on accelerometer data.

    configuration:
      --config CONFIG       Path to YAML config file. Overrides command-line
                            defaults (default: None)

    required arguments:
      --data_path DATA_PATH
                            Directory containing split HDF5 files from Step T1
                            (must contain 10s_train.h5, 10s_val.h5,
                            10s_test_complete.h5)
      --output_dir OUTPUT_DIR
                            Directory to save trained checkpoints and logs

    model arguments:
      --model MODEL         Model architecture name. Use "CHAP" for the CNN-BiLSTM
                            model, or "vit_base_patch16" / "vit_small_patch16" /
                            "vit_tiny_patch16" for ViT variants (default:
                            vit_base_patch16)
      --checkpoint CHECKPOINT
                            Path to pre-trained weights to initialize from.
                            Recommended: use MSSE_2021_pt/pre-trained-models-pt/
                            CHAP_ALL_ADULTS.pth for CHAP models (default: None)
      --remark REMARK       Experiment name used in log filenames and output
                            directories (default: Debug)
      --input_size INPUT_SIZE
                            Model input size in timesteps (default: 4200)
      --patch_size PATCH_SIZE
                            Patch size for ViT models (default: 100)
      --patch_nvar PATCH_NVAR
                            Number of variables per patch (default: 1)
      --in_chans IN_CHANS   Number of input channels (default: 3)
      --use_pos_embed       Use positional embeddings (default: False)
      --no_use_pos_embed    Disable positional embeddings
      --use_rope            Use Rotary Position Embedding (default: False)
      --patch_emb PATCH_EMB
                            Patch embedding type (default: vit)
      --drop_path_rate DROP_PATH_RATE
                            Stochastic depth / drop path rate (default: 0.1)
      --num_attn_layer NUM_ATTN_LAYER
                            Number of attention layers in AttentionProbeModel
                            (default: 2)

    optimizer arguments:
      --blr BLR             Base learning rate. Actual lr = blr * total_batch_size
                            / 256 (default: 5e-4)
      --lr LR               Absolute learning rate. If set, overrides --blr
                            (default: None)
      --min_lr MIN_LR       Lower learning rate bound for cosine scheduler
                            (default: 1e-6)
      --layer_decay LAYER_DECAY
                            Layer-wise learning rate decay factor (default: 0.75)
      --warmup_epochs WARMUP_EPOCHS
                            Number of epochs for linear learning rate warmup
                            (default: 2)
      --weight_decay WEIGHT_DECAY
                            Weight decay / L2 regularization (default: 0.05)
      --clip_grad CLIP_GRAD
                            Gradient norm clipping threshold. None = no clipping
                            (default: None)

    training arguments:
      --epochs EPOCHS       Total number of training epochs (default: 20)
      --batch_size BATCH_SIZE
                            Batch size per GPU. Effective batch size =
                            batch_size * accum_iter * num_gpus (default: 64)
      --accum_iter ACCUM_ITER
                            Gradient accumulation steps. Increase to simulate
                            larger batch sizes under memory constraints (default: 1)
      --use_data_aug USE_DATA_AUG
                            Enable data augmentation: 1 = on, 0 = off (default: 1)
      --subset_ratio SUBSET_RATIO
                            Fraction of training data to use. 1.0 = all data
                            (default: 1.0)
      --use_focal_loss      Use focal loss instead of BCEWithLogitsLoss for
                            class-imbalanced data (default: False)
      --pos_weight POS_WEIGHT
                            Positive class weight for BCE loss (default: 1.0)
      --resume RESUME       Path to checkpoint to resume training from
                            (default: None)
      --start_epoch START_EPOCH
                            Epoch number to resume from (default: 0)

    dataset arguments:

      --nb_classes NB_CLASSES
                            Number of output classes (default: 2)

    runtime arguments:
      --device DEVICE       Device: "cpu" or "cuda" (default: cuda)
      --seed SEED           Random seed for reproducibility (default: 0)
      --num_workers NUM_WORKERS
                            Number of parallel data loading workers (default: 4)
      --pin_mem             Pin CPU memory in DataLoader (default: True)
      --no_pin_mem          Disable pinned memory
      --log_dir LOG_DIR     Directory for TensorBoard logs (default:
                            /niddk-data-central/log)
      --dist_eval           Use distributed evaluation during training
                            (default: False)

    distributed training (advanced):
      --world_size WORLD_SIZE
                            Number of distributed processes (default: 1)
      --local_rank LOCAL_RANK
                            Local rank for distributed training (default: -1)
      --dist_on_itp         Use ITP distributed backend (default: False)
      --dist_url DIST_URL   URL for distributed training setup (default: env://)

</details>

**Step T3 — Evaluate / generate per-subject predictions on labeled splits.**
Once you have a trained checkpoint and labeled splits from Step T1, run the model
in evaluation mode to get balanced accuracy / F1 / confusion matrix on each split
*and* per-subject prediction CSVs:

```bash
cd CHAP2
python -m main_finetune \
  --data_path /path/to/your_split_data \
  --model CHAP \
  --eval SUBMIT_RESULT/SOL_W/CHAP_FT/checkpoint-submit.pth \
  --make_prediction \
  --prediction_dir /path/to/save_predictions \
  --batch_size 16 \
  --device cpu \
  --num_workers 0
```

If you changed default model parameters during training (e.g., `--input_size`,
`--patch_size`), set the same values here. Use this path when you have ground
truth and want metrics; for prediction-only on raw CHAP1 data without labels,
use [`make_predictions.py`](#step-2--predict-run-a-trained-model) instead.

<details>
<summary>Complete <code>main_finetune</code> evaluation / prediction options</summary>

    usage: main_finetune [-h] [--data_path DATA_PATH]
                         [--model MODEL] [--eval EVAL]
                         [--make_prediction] [--prediction_dir PREDICTION_DIR]
                         [--batch_size BATCH_SIZE]
                         [--device DEVICE] [--num_workers NUM_WORKERS]
                         [--input_size INPUT_SIZE] [--patch_size PATCH_SIZE]
                         [--patch_nvar PATCH_NVAR] [--in_chans IN_CHANS]
                         [--nb_classes NB_CLASSES] [--seed SEED]
                         [--pin_mem | --no_pin_mem]
                         [--use_pos_embed | --no_use_pos_embed]
                         [--use_rope] [--patch_emb PATCH_EMB]
                         [--num_attn_layer NUM_ATTN_LAYER]
                         [--subject_level_analysis]

    Generate predictions with a trained model.

    required arguments:
      --data_path DATA_PATH
                            Directory containing split HDF5 files from Step T1
                            (must contain 10s_train.h5, 10s_val.h5,
                            10s_test_complete.h5)
      --eval EVAL           Path to a trained model checkpoint (.pth file).
                            See "Submitted Weights" for provided weights
      --make_prediction     Flag to enable per-subject CSV output (no value needed)

    optional arguments:
      -h, --help            show this help message and exit
      --model MODEL         Model architecture name. Use "CHAP" for the CNN-BiLSTM
                            model, or "vit_base_patch16" / "vit_small_patch16" /
                            "vit_tiny_patch16" for ViT variants (default:
                            vit_base_patch16)
      --prediction_dir PREDICTION_DIR
                            Directory to save prediction CSV files (default: None)
      --batch_size BATCH_SIZE
                            Inference batch size (default: 64). Reduce if you get
                            out-of-memory errors
      --device DEVICE       Device for inference: "cpu" or "cuda" (default: cuda)
      --num_workers NUM_WORKERS
                            Number of parallel data loading workers. Use 0 for
                            debugging, 4 for speed (default: 4)
      --input_size INPUT_SIZE
                            Model input size in timesteps (default: 4200, i.e.
                            42 windows x 100 samples)
      --patch_size PATCH_SIZE
                            Patch size for ViT models in timesteps (default: 100)
      --patch_nvar PATCH_NVAR
                            Number of variables per patch (default: 1)
      --in_chans IN_CHANS   Number of input channels / accelerometer axes
                            (default: 3)
      --nb_classes NB_CLASSES
                            Number of output classes (default: 2)
      --seed SEED           Random seed for reproducibility (default: 0)
      --pin_mem             Pin CPU memory in DataLoader (default: True)
      --no_pin_mem          Disable pinned memory
      --use_pos_embed       Use positional embeddings (default: False)
      --no_use_pos_embed    Disable positional embeddings
      --use_rope            Use Rotary Position Embedding (default: False)
      --patch_emb PATCH_EMB
                            Patch embedding type (default: vit)
      --num_attn_layer NUM_ATTN_LAYER
                            Number of attention layers in AttentionProbeModel
                            (default: 2)
      --subject_level_analysis
                            Enable per-subject analysis in output (default: False)

</details>

Ready-to-edit training scripts with recommended settings are in `CHAP2/script/`:

```bash
bash script/chap_ft_sol.sh        # Finetune CHAP on SOL/PASOS wrist
bash script/chap_ft_iwatch.sh     # Finetune CHAP on iWatch
bash script/iwatch_vit.sh         # Finetune ViT on iWatch
```

> Edit the `.sh` files to change dataset paths and hyperparameters for your setup.


Reference
=========

Data Formats
------------

You only need this if you're preparing data by hand or debugging. Understanding
the formats at each step helps when preparing your own data.

### Source data (output of CHAP1 preprocessing)

After [Step 1](#step-1--preprocess-raw-actigraph-csv--hdf5), each subject's data
is a folder of daily HDF5 files:

```
your_preprocessed_dir/
├── subject_001/          # one folder per subject
│   ├── 2023-01-01.h5     # one file per day
│   ├── 2023-01-02.h5
│   └── ...
├── subject_002/
│   └── ...
```

Each daily `.h5` file contains 10-second windows of accelerometer data:

| Field | Shape | Description |
|-------|-------|-------------|
| `data` | `(N, 100, 3)` | Accelerometer readings (x, y, z) at 10 Hz. Each row = 100 samples over 10 seconds, 3 axes. |
| `label` | `(N,)` | `0` = sitting, `1` = not sitting, `-1` = unknown. |
| `time` | `(N,)` | Unix timestamp (seconds since 1970) for each window. |
| `non_wear` | `(N,)` | `0` = device was worn, `1` = device was not worn. |
| `sleeping` | `(N,)` | `0` = awake, `1` = sleeping. |

For example, `DEMO/CHAP1_preprocess_demo/demo_subject/2018-06-25.h5` contains
358 ten-second windows from one day:

```
data:     shape=(358, 100, 3)   e.g. [ 0.0055, -0.9794, -0.0715]  (x, y, z in g)
label:    shape=(358,)          e.g. [1, 1, 1, 1, ...]             (1=not sitting)
time:     shape=(358,)          e.g. [1.5299e+09, ...]             (Unix timestamp)
non_wear: shape=(358,)          e.g. [0, 0, 0, ...]                (0=device worn)
sleeping: shape=(358,)          e.g. [0, 0, 0, ...]                (0=awake)
```

### Train/val/test split file

To split subjects into train, validation, and test sets, you provide a CSV file
with two columns:

```csv
subject_id,split
subject_001,train
subject_002,train
subject_003,validation
subject_004,test
```

Each row assigns one subject to a split. The `subject_id` must match the folder
names in your preprocessed directory.

### Split HDF5 files (output of `create_dataset_split.py`, used for training)

After running `create_dataset_split.py`, the 10-second windows are grouped into
longer segments (default: 42 consecutive windows = ~7 minutes) and saved as:

```
output_dir/
├── 10s_train.h5
├── 10s_val.h5
└── 10s_test_complete.h5
```

Each file contains:

| Field | Shape | Description |
|-------|-------|-------------|
| `x` | `(N, 42, 100, 3)` | Accelerometer data. N samples, each is 42 consecutive 10-second windows. |
| `y` | `(N, 42)` | Labels for each 10-second window within the ~7-minute segment. |
| `timestamp` | `(N, 42)` | Timestamps for each window. |
| `subject_id` | `(N,)` | Which subject this sample belongs to. |
| `std` | `(N, 42)` | Standard deviation of accelerometer signal (used for sampling). |

For example, `DEMO/demo_output/10s_train.h5` contains 8 samples:

```
x:          shape=(8, 42, 100, 3)   <- 8 samples, each 42 windows x 100 timesteps x 3 axes
y:          shape=(8, 42)           <- 8 samples, each with 42 labels
subject_id: shape=(8,)             <- e.g. ["demo_subject", "demo_subject", ...]
timestamp:  shape=(8, 42)          <- Unix timestamps
```


Repository Structure
--------------------

```
CHAP2/                          # Main pipeline
├── DEMO/                       # Self-contained demo data and outputs
│   ├── CHAP1_preprocess_demo/  #   Sample preprocessed data (1 subject, 1 day)
│   ├── demo_output/            #   Windowed HDF5 splits (only used for training)
│   └── demo_prediction/        #   Prediction CSVs (output of make_predictions.py)
├── make_predictions.py         # Prediction-only entry point (reads CHAP1 dirs directly)
├── create_dataset_split.py     # Training only: create train/val/test HDF5 splits
├── main_finetune.py            # Training / evaluation entry point
├── chap_model.py               # CHAP model definitions (CNN-BiLSTM)
├── models_vit.py               # Vision Transformer model definitions (CHAP-ViT)
├── vision_transformer.py       # ViT with Rotary Position Embedding (RoPE)
├── engine_finetune.py          # Training and evaluation loops
├── example_models.py           # Standalone example: model loading and inference
├── requirements.txt            # Python dependencies
├── util/                       # Utilities (data loading, learning rate, loss, etc.)
├── script/                     # Shell scripts for training and demo
│   ├── make_prediction_cpu_demo.sh     # Demo script (CPU, single subject)
│   ├── chap_ft_sol.sh                  # SOL dataset CHAP-FT finetuning
│   ├── chap_ft_iwatch.sh              # iWatch CHAP-FT finetuning
│   ├── iwatch_vit.sh                  # iWatch CHAP-ViT finetuning
│   └── chap_scratch_sol.sh            # SOL training from scratch
└── SUBMIT_RESULT/              # Trained checkpoints and predictions
    ├── SOL_W/CHAP_FT/                 # SOL/PASOS Wrist (fine-tuned)
    ├── iWatch_H/                       # iWatch Hip models
    │   ├── CHAP-FT/checkpoint/
    │   └── CHAP-ZS/checkpoint/
    └── iWatch_W/                       # iWatch Wrist models
        ├── CHAP-FT/checkpoint/
        └── CHAP-ZS/checkpoint/

MSSE_2021_pt/               # PyTorch port of CNN/BiLSTM baselines (CHAP 1.0) + preprocessing
support_files/              # Validation data and participant-level agreement files
```


Submitted Weights (SUBMIT_RESULT)
----------------------------------

Each subdirectory in `CHAP2/SUBMIT_RESULT/` contains a
`checkpoint/checkpoint-submit.pth` and per-subject prediction CSVs.

All weights are CHAP models (CNN-BiLSTM architecture) initialized from CHAP 1.0
pre-trained weights (ACT + AUSDIAB hip cohorts, stored in
`MSSE_2021_pt/pre-trained-models-pt/CHAP_ALL_ADULTS.pth`). CHAP-FT checkpoints are
finetuned starting from their respective CHAP-ZS weights.

| Weight | Finetuned On | Notes |
|--------|--------------|-------|
| `iWatch_H/CHAP-ZS` | — (zero-shot) | Directly applies the pre-trained hip model to iWatch hip data without any finetuning. |
| `iWatch_H/CHAP-FT` | iWatch Hip | Finetuned from `iWatch_H/CHAP-ZS` on iWatch hip data (40 epochs, lr=1e-3). See `script/chap_ft_iwatch.sh`. |
| `iWatch_W/CHAP-ZS` | — (zero-shot) | Directly applies the pre-trained hip model to iWatch wrist data without any finetuning. |
| `iWatch_W/CHAP-FT` | iWatch Wrist | Finetuned from `iWatch_W/CHAP-ZS` on iWatch wrist data (40 epochs, lr=1e-3). See `script/chap_ft_iwatch.sh`. |
| `SOL_W/CHAP_FT` | SOL/PASOS Wrist | Finetuned on SOL/PASOS wrist data (10 epochs, lr=1e-3). See `script/chap_ft_sol.sh`. |

See `CHAP2/example_models.py` for a standalone example of model instantiation,
weight loading, and inference.


Related Publications
--------------------
- **JMPB 2021:** *Application of Convolutional Neural Network Algorithms for Advancing Sedentary and Activity Bout Classification* — [DOI](https://doi.org/10.1123/jmpb.2020-0016) | [Paper](https://adalabucsd.github.io/papers/2021_JMPB_CNN.pdf)
- **CHAP (MSSE 2021):** *The CNN Hip Accelerometer Posture (CHAP) Method for Classifying Sitting Patterns from Hip Accelerometers: A Validation Study* — [DOI](https://doi.org/10.1249/MSS.0000000000002705)
- **CHAP-child (IJBNPA 2022):** *CHAP-child: An Open Source Method for Estimating Sit-to-Stand Transitions and Sedentary Bout Patterns from Hip Accelerometers Among Children* — [DOI](https://doi.org/10.1186/s12966-022-01349-2)
- **CHAP-Adult (JMPB 2022):** *CHAP-Adult: A Reliable and Valid Algorithm to Classify Sitting and Measure Sitting Patterns Using Data From Hip-Worn Accelerometers in Adults Aged 35+* — [DOI](https://doi.org/10.1123/jmpb.2021-0062)
- **ISCOLE (IJO 2023):** *Low Movement, Deep-learned Sitting Patterns, and Sedentary Behavior in the International Study of Childhood Obesity, Lifestyle and the Environment (ISCOLE)* — [DOI](https://doi.org/10.1038/s41366-023-01364-8)
- **OPACH (JAHA 2024):** *Prospective Associations of Accelerometer-Measured Machine-Learned Sedentary Behavior With Death Among Older Women: The OPACH Study* — [DOI](https://doi.org/10.1161/JAHA.123.031156)
- **JREHD 2025:** *Movement- and Posture-based Measures of Sedentary Patterns and Associations with Metabolic Syndrome in Hispanic/Latino and non-Hispanic Adults* — [DOI](https://doi.org/10.1007/s40615-024-02114-w)


Acknowledgement
---------------
This work was supported by grant number R01DK114945 from the National Institute of Diabetes and Digestive and Kidney Diseases, and by grant number 1R01HL168535 from the National Heart, Lung and Blood Institute. It was also supported in part by a Hellman Fellowship, an NSF CAREER Award under award number 1942724, and a gift from VMware. The content is solely the responsibility of the authors and does not necessarily represent the views of any of these organizations. We thank the members of UC San Diego's Database Lab and Center for Networked Systems for their feedback on this work.
