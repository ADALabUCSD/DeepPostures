#!/bin/bash
# =============================================================================
# Demo: prediction on a single subject of CHAP1-preprocessed wrist data (CPU).
#
# Layout:
#   DEMO/CHAP1_preprocess_demo/<subject_id>/<YYYY-MM-DD>.h5   <- CHAP1 output
#   DEMO/demo_prediction/<subject_id>.csv                     <- this script's output
#
# No train/val/test split is needed for prediction. Use create_dataset_split.py
# only when you want to (re)train; see chap_ft_sol.sh / chap_ft_iwatch.sh.
# =============================================================================

# Key parameters:
#   --data_dir        CHAP1-preprocessed dir with one subdir per subject
#   --checkpoint      Trained CHAP weights (.pth)
#   --prediction_dir  Where per-subject prediction CSVs are written
#   --batch_size      Inference batch size (reduce if OOM)
#   --device          cpu or cuda (default: cuda)
#
# Available checkpoints in SUBMIT_RESULT/:
#   SUBMIT_RESULT/SOL_W/CHAP_FT/checkpoint-submit.pth                (SOL wrist, fine-tuned)
#   SUBMIT_RESULT/iWatch_W/CHAP-FT/checkpoint/checkpoint-submit.pth  (iWatch wrist, fine-tuned)
#   SUBMIT_RESULT/iWatch_W/CHAP-ZS/checkpoint/checkpoint-submit.pth  (iWatch wrist, zero-shot)
#   SUBMIT_RESULT/iWatch_H/CHAP-FT/checkpoint/checkpoint-submit.pth  (iWatch hip, fine-tuned)
#   SUBMIT_RESULT/iWatch_H/CHAP-ZS/checkpoint/checkpoint-submit.pth  (iWatch hip, zero-shot)

python -m make_predictions \
  --data_dir       "DEMO/CHAP1_preprocess_demo" \
  --checkpoint     "SUBMIT_RESULT/SOL_W/CHAP_FT/checkpoint-submit.pth" \
  --prediction_dir "DEMO/demo_prediction" \
  --batch_size 5 \
  --device cpu
