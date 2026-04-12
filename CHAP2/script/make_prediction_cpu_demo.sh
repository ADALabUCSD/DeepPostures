#!/bin/bash
# =============================================================================
# Demo: End-to-end prediction pipeline on a single subject (CPU-only)
#
# All demo inputs/outputs live under DEMO/:
#   DEMO/CHAP1_preprocess_demo/   <- input: preprocessed daily .h5 from CHAP1
#   DEMO/demo_output/             <- step 2 output: windowed HDF5 splits
#   DEMO/demo_prediction/         <- step 3 output: per-subject predictions
# =============================================================================

# -- STEP 1: Preprocessed Dataset (CHAP1.0 Preprocessing Output) -------------
# Expected: DEMO/CHAP1_preprocess_demo/<YYYY-MM-DD>.h5
# Each .h5 contains keys: data, label, non_wear, sleeping, time
# (Produced by the CHAP1 preprocessing pipeline — already provided for demo)

# -- STEP 2: Create train/val/test splits ------------------------------------
# Reads from DEMO/CHAP1_preprocess_demo/, writes windowed HDF5 files to
# DEMO/demo_output/10s_{train,val,test_complete}.h5
#
# For full run with multiple subjects, use:
#   python create_dataset_split.py \
#     --data_dir <pre_processed_dir>  \   # dir with per-subject subdirs from CHAP1
#     --split_csv <split.csv>         \   # CSV: subject_id, split (train/validation/test)
#     --output_dir <output_dir>           # writes 10s_train.h5, 10s_val.h5, 10s_test_complete.h5

python create_dataset_split.py --demo

# -- STEP 3: Make Prediction (SOL_CHAP_FT_Wrist) ------------------------------
# Key parameters:
#   --data_path    Dir containing 10s_{train,val,test_complete}.h5 from Step 2
#   --model        Model architecture. Options: CHAP, vit_base_patch16, ...
#   --eval         Path to a trained checkpoint (.pth) to load for inference
#   --make_prediction      Enable prediction mode (no training)
#   --prediction_dir       Where to save per-subject CSV predictions
#   --batch_size   Samples per batch (reduce if OOM)
#   --device       cpu or cuda (default: cuda)
#   --num_workers  DataLoader workers (0 = main process only, good for debugging)
#
# Available checkpoints in SUBMIT_RESULT/:
#   SUBMIT_RESULT/SOL_W/CHAP_FT/checkpoint-submit.pth          (SOL wrist, fine-tuned)
#   SUBMIT_RESULT/iWatch_W/CHAP-FT/checkpoint/checkpoint-submit.pth  (iWatch wrist, fine-tuned)
#   SUBMIT_RESULT/iWatch_W/CHAP-ZS/checkpoint/checkpoint-submit.pth  (iWatch wrist, zero-shot)
#   SUBMIT_RESULT/iWatch_H/CHAP-FT/checkpoint/checkpoint-submit.pth  (iWatch hip, fine-tuned)
#   SUBMIT_RESULT/iWatch_H/CHAP-ZS/checkpoint/checkpoint-submit.pth  (iWatch hip, zero-shot)

python -m main_finetune \
  --data_path "DEMO/demo_output" \
  --model CHAP \
  --eval "SUBMIT_RESULT/SOL_W/CHAP_FT/checkpoint-submit.pth" \
  --make_prediction \
  --prediction_dir "DEMO/demo_prediction" \
  --batch_size 5 \
  --device cpu \
  --num_workers 0
