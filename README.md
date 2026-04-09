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

For example, `DEMO/CHAP1_preprocess_demo/2018-06-25.h5` contains 358 ten-second
windows from one day:

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

### Split HDF5 files (output of Step 2: `create_dataset_split.py`)

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
│   ├── demo_output/            #   Windowed HDF5 splits (created by Step 2)
│   └── demo_prediction/        #   Prediction CSVs (created by Step 3)
├── create_dataset_split.py     # Step 2: Create train/val/test HDF5 splits
├── main_finetune.py            # Training and prediction entry point
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

A toy example is included so you can verify your setup works. It runs the full
pipeline on a single subject of wrist accelerometer data — **no GPU required**.

1. `cd` into the `CHAP2/` folder:

```bash
cd CHAP2
```

2. Run the demo script:

```bash
bash script/make_prediction_cpu_demo.sh
```

This script does three things automatically:

| Step | What it does | Input | Output |
|------|-------------|-------|--------|
| 1 | (Already done) Sample data is in `DEMO/CHAP1_preprocess_demo/` | — | — |
| 2 | Splits the data into train/val/test sets | `DEMO/CHAP1_preprocess_demo/` | `DEMO/demo_output/10s_{train,val,test_complete}.h5` |
| 3 | Loads a trained model and generates predictions | `DEMO/demo_output/` + checkpoint | `DEMO/demo_prediction/` (per-subject CSV files) |

3. Check the results in `DEMO/demo_prediction/`.

See `script/make_prediction_cpu_demo.sh` for detailed comments explaining every
parameter — it serves as a reference for building your own commands.


Usage — Running on Your Own Data
---------------------------------

The demo above shows the full pipeline. To run on your own data, replace the demo
paths with your own. See `CHAP2/example_models.py` for a standalone example of
model instantiation, weight loading, and inference.

### Step 1: Prepare your data (CHAP1 preprocessing)

Run the CHAP1 preprocessing pipeline on your raw accelerometer data to produce
daily HDF5 files. The output should follow the folder structure shown in "What
the Data Looks Like" above.

### Step 2: Create train/val/test splits

Prepare a split CSV (see format in "What the Data Looks Like"), then run:

```bash
python create_dataset_split.py \
  --data_dir /path/to/your_preprocessed_dir \
  --split_csv /path/to/your_split.csv \
  --output_dir /path/to/output
```

### Step 3: Generate predictions with a trained model

```bash
python -m main_finetune \
  --data_path /path/to/output \
  --model CHAP \
  --eval SUBMIT_RESULT/SOL_W/CHAP_FT/checkpoint-submit.pth \
  --make_prediction \
  --prediction_dir /path/to/save_predictions \
  --ds_name iwatch \
  --batch_size 16 \
  --device cpu \
  --num_workers 0
```

**What each flag means:**

| Flag | What it controls | What to put |
|------|-----------------|-------------|
| `--data_path` | Where your split HDF5 files are (from Step 2) | A folder containing `10s_train.h5`, `10s_val.h5`, `10s_test_complete.h5` |
| `--model` | Which model architecture to use | `CHAP` for CNN-BiLSTM (recommended) |
| `--eval` | Path to a trained model checkpoint | See "Available Checkpoints" below |
| `--make_prediction` | Tells the program to generate predictions (not train) | Just include this flag, no value needed |
| `--prediction_dir` | Where to save the prediction CSV files | Any folder path you choose |
| `--ds_name` | Dataset loader to use | `iwatch` |
| `--batch_size` | How many samples to process at once | `16` is a good default; reduce if you get memory errors |
| `--device` | Run on CPU or GPU | `cpu` or `cuda` (use `cuda` if you have a GPU — much faster) |
| `--num_workers` | Parallel data loading threads | `0` for debugging, `4` for speed |

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
provided checkpoints:

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

**Additional flags for training:**

| Flag | What it controls | Default |
|------|-----------------|---------|
| `--checkpoint` | Pre-trained weights to start from (recommended) | None |
| `--epochs` | Number of training passes over the data | 20 |
| `--blr` | Base learning rate | 5e-4 |
| `--weight_decay` | Regularization strength | 5e-2 |
| `--warmup_epochs` | Gradual learning rate warmup period | 2 |
| `--use_data_aug` | Data augmentation (`1` = on, `0` = off) | 1 |
| `--output_dir` | Where to save trained checkpoints | — |
| `--remark` | Name for this experiment (used in logs and filenames) | Debug |
| `--resume` | Resume training from a saved checkpoint | — |

Training scripts with recommended settings are in `CHAP2/script/`:

```bash
bash script/chap_ft_sol.sh        # Finetune CHAP on SOL/PASOS wrist
bash script/chap_ft_iwatch.sh     # Finetune CHAP on iWatch
bash script/iwatch_vit.sh         # Finetune ViT on iWatch
```

> Edit the `.sh` files to change dataset paths and hyperparameters for your setup.

For the full list of all parameters, run `python -m main_finetune --help`.


Related Publications
--------------------
- **CHAP (MSSE 2021):** *The CNN Hip Accelerometer Posture (CHAP) Method for Classifying Sitting Patterns from Hip Accelerometers: A Validation Study in Older Adults*
- **JMPB 2021:** *Application of Convolutional Neural Network Algorithms for Advancing Sedentary and Activity Bout Classification* — [DOI](https://doi.org/10.1123/jmpb.2020-0016) | [Paper](https://adalabucsd.github.io/papers/2021_JMPB_CNN.pdf)


Acknowledgement
---------------
This work was supported by grant number R01DK114945 from the National Institute of Diabetes and Digestive and Kidney Diseases. It was also supported in part by a Hellman Fellowship, an NSF CAREER Award under award number 1942724, and a gift from VMware. The content is solely the responsibility of the authors and does not necessarily represent the views of any of these organizations. We thank the members of UC San Diego's Database Lab and Center for Networked Systems for their feedback on this work.