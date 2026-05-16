CHAP 2.0 — Finetuning for Wrist & Hip Accelerometer Posture Classification
============================================================================

This repository contains the **PyTorch** codebase for classifying sedentary and activity postures from accelerometer data. It includes all models from the paper:

- **CHAP-ZS**: Zero-shot CHAP model (no finetuning)
- **CHAP-FT**: Finetuned CHAP model for posture classification
- **CHAP-ViT**: Vision Transformer-based model variants (ViT-base, ViT-small, ViT-tiny)

All models support finetuning on both hip and wrist (iWatch) accelerometer datasets, including SOL/PASOS.

For the legacy TensorFlow code and earlier publications, see the `master` branch.


What the Data Looks Like
-------------------------

Understanding the data formats at each step helps when preparing your own data.

### Source data (output of CHAP1 preprocessing)

After CHAP1 preprocessing, each subject's data is organized in a folder of daily HDF5 files:

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

MSSE_2021_pt/               # PyTorch port of CNN/BiLSTM baselines (CHAP 1.0)
support_files/              # Validation data and participant-level agreement files
```


Submitted Weights (SUBMIT_RESULT)
----------------------------------

Each subdirectory in `CHAP2/SUBMIT_RESULT/` contains a `checkpoint/checkpoint-submit.pth` and per-subject prediction CSVs. Below is a summary of each weight:

All weights are CHAP models (CNN-BiLSTM architecture) initialized from CHAP 1.0 pre-trained weights (ACT + AUSDIAB hip cohorts, stored in `MSSE_2021_pt/pre-trained-models-pt/`). CHAP-FT checkpoints are finetuned starting from their respective CHAP-ZS weights.

| Weight | Finetuned On | Notes |
|--------|--------------|-------|
| `iWatch_H/CHAP-ZS` | — (zero-shot) | Directly applies the pre-trained hip model to iWatch hip data without any finetuning. |
| `iWatch_H/CHAP-FT` | iWatch Hip | Finetuned from `iWatch_H/CHAP-ZS` on iWatch hip data (40 epochs, lr=1e-3). See `script/chap_ft_iwatch.sh`. |
| `iWatch_W/CHAP-ZS` | — (zero-shot) | Directly applies the pre-trained hip model to iWatch wrist data without any finetuning. |
| `iWatch_W/CHAP-FT` | iWatch Wrist | Finetuned from `iWatch_W/CHAP-ZS` on iWatch wrist data (40 epochs, lr=1e-3). See `script/chap_ft_iwatch.sh`. |
| `SOL_W/CHAP_FT` | SOL/PASOS Wrist | Finetuned on SOL/PASOS wrist data (10 epochs, lr=1e-3). See `script/chap_ft_sol.sh`. |


Pre-Requisites
--------------
- Python 3.8+
- [Conda](https://github.com/conda-forge/miniforge/releases/latest) (Miniforge recommended)
- A GPU machine is strongly recommended for training.

Environment Setup:

```bash
conda create -n chap python=3.10 -y
conda activate chap

# From repo root:
pip install -r CHAP2/requirements.txt

# Or from inside CHAP2/:
pip install -r requirements.txt
```


Quick Start (CPU Demo)
---------------------

A toy example is included so you can verify your setup works. It runs prediction
on a single subject of wrist accelerometer data — **no GPU required**.

1. `cd` into the `CHAP2/` folder:

```bash
cd CHAP2
```

2. Run the demo script:

```bash
bash script/make_prediction_cpu_demo.sh
```

This loads the SOL/PASOS wrist checkpoint and runs prediction directly on the
bundled CHAP1-preprocessed data:

| Input | Output |
|-------|--------|
| `DEMO/CHAP1_preprocess_demo/<subject_id>/<YYYY-MM-DD>.h5` + `SUBMIT_RESULT/SOL_W/CHAP_FT/checkpoint-submit.pth` | `DEMO/demo_prediction/<subject_id>.csv` |

3. Check the results in `DEMO/demo_prediction/`.

> Prediction does **not** require running `create_dataset_split.py` and does
> not need train/val/test files — that step is only for (re)training. See
> `script/make_prediction_cpu_demo.sh` for parameter details.


Usage — Running on Your Own Data
---------------------------------

There are two paths depending on what you want:

| Goal | Use |
|------|-----|
| **Prediction-only** on CHAP1-preprocessed data, **with or without labels** — no splits, no metrics | `make_predictions.py` (Step 2 below) |
| Full **train / validate / test** loop, or per-subject prediction **with metrics** on labeled splits | `create_dataset_split.py` + `main_finetune.py` (see "Finetuning Your Own Model" below) |

See `CHAP2/example_models.py` for a standalone example of model instantiation,
weight loading, and inference.

### Step 1: Prepare your data (CHAP1 preprocessing)

Run the CHAP1 preprocessing pipeline on your raw accelerometer data to produce
daily HDF5 files in the `<dir>/<subject_id>/<YYYY-MM-DD>.h5` layout shown in
"What the Data Looks Like" above.

### Step 2 (prediction-only): Generate predictions with a trained model

For prediction-only, pass the CHAP1-preprocessed dir directly to
`make_predictions.py` — no train/val/test split is required, and **labels in the
input are optional**:

```bash
python -m make_predictions \
  --data_dir       /path/to/your_preprocessed_dir \
  --checkpoint     SUBMIT_RESULT/SOL_W/CHAP_FT/checkpoint-submit.pth \
  --prediction_dir /path/to/save_predictions \
  --batch_size 64 \
  --device cuda
```

One CSV per subject is written to `--prediction_dir`. Columns are
`segment, timestamp, prediction`; a `label` column is appended only if the
input has any labeled windows. If you need balanced accuracy / F1 / confusion
matrix on labeled data, use the `main_finetune --make_prediction` flow under
"Finetuning Your Own Model" instead.

Complete usage:

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

### Available Checkpoints

These trained model weights are included in `SUBMIT_RESULT/`:

| Checkpoint path | Trained on | Description |
|----------------|-----------|-------------|
| `SUBMIT_RESULT/SOL_W/CHAP_FT/checkpoint-submit.pth` | SOL/PASOS wrist | Fine-tuned on SOL wrist data |
| `SUBMIT_RESULT/iWatch_W/CHAP-FT/checkpoint/checkpoint-submit.pth` | iWatch wrist | Fine-tuned on iWatch wrist data |
| `SUBMIT_RESULT/iWatch_W/CHAP-ZS/checkpoint/checkpoint-submit.pth` | — (zero-shot) | Pre-trained model, no finetuning |
| `SUBMIT_RESULT/iWatch_H/CHAP-FT/checkpoint/checkpoint-submit.pth` | iWatch hip | Fine-tuned on iWatch hip data |
| `SUBMIT_RESULT/iWatch_H/CHAP-ZS/checkpoint/checkpoint-submit.pth` | — (zero-shot) | Pre-trained model, no finetuning |

All CHAP weights start from the CHAP 1.0 pre-trained model (`MSSE_2021_pt/pre-trained-models-pt/CHAP_ALL_ADULTS.pth`), originally trained on ACT + AUSDIAB hip cohorts. CHAP-FT checkpoints are further finetuned on the target dataset.


Finetuning Your Own Model
--------------------------

If you want to train (finetune) a model on your own dataset instead of using the
provided checkpoints, training requires pre-built train/val/test HDF5 splits.

**Step T1 — Build splits.** Prepare a split CSV (see "Train/val/test split file"
above), then run:

```bash
python create_dataset_split.py \
  --data_dir /path/to/your_preprocessed_dir \
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

Complete usage details of `main_finetune` for training with all configuration options:

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

**Step T3 — Evaluate / generate per-subject predictions on labeled splits.**
Once you have a trained checkpoint and labeled train/val/test splits from
Step T1, you can run the model in evaluation mode through `main_finetune` to
get balanced accuracy / F1 / confusion matrix on each split *and* per-subject
prediction CSVs (this is the original full-loop workflow):

```bash
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
use `make_predictions.py` (Step 2 above) instead.

Complete usage details of `main_finetune` for evaluation / prediction:

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
                            See "Available Checkpoints" above for provided weights
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

Training scripts with recommended settings are in `CHAP2/script/`:

```bash
bash script/chap_ft_sol.sh        # Finetune CHAP on SOL/PASOS wrist
bash script/chap_ft_iwatch.sh     # Finetune CHAP on iWatch
bash script/iwatch_vit.sh         # Finetune ViT on iWatch
```

> Edit the `.sh` files to change dataset paths and hyperparameters for your setup.


Related Publications
--------------------
- **CHAP (MSSE 2021):** *The CNN Hip Accelerometer Posture (CHAP) Method for Classifying Sitting Patterns from Hip Accelerometers: A Validation Study in Older Adults*
- **JMPB 2021:** *Application of Convolutional Neural Network Algorithms for Advancing Sedentary and Activity Bout Classification* — [DOI](https://doi.org/10.1123/jmpb.2020-0016) | [Paper](https://adalabucsd.github.io/papers/2021_JMPB_CNN.pdf)


Acknowledgement
---------------
This work was supported by grant number R01DK114945 from the National Institute of Diabetes and Digestive and Kidney Diseases. It was also supported in part by a Hellman Fellowship, an NSF CAREER Award under award number 1942724, and a gift from VMware. The content is solely the responsibility of the authors and does not necessarily represent the views of any of these organizations. We thank the members of UC San Diego's Database Lab and Center for Networked Systems for their feedback on this work.