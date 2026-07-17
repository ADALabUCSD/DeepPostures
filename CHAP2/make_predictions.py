"""
Generate per-subject prediction CSVs from CHAP-preprocessed daily HDF5 data.

Reads daily HDF5 output directly:
    <data_dir>/<subject_id>/<YYYY-MM-DD>.h5

Each daily .h5 has keys:
    data      (N, 100, 3)   accelerometer (x, y, z) at 10 Hz
    label     (N,)          0=sitting, 1=not-sitting, -1=unknown
    time      (N,)          Unix timestamp per 10s window
    non_wear  (N,)          1 = device off
    sleeping  (N,)          1 = asleep

For each subject, contiguous awake/worn 10s windows are grouped into chunks of
--window_size (default 42 = ~7 minutes), the model is run on each chunk, and
one CSV is written per subject:
    segment, timestamp, prediction[, label]

The `label` column is included only if the input has any labeled windows;
otherwise prediction-only output is written.

This script does NOT require running create_dataset_split.py first, and does
NOT need train/validation splits.
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from tqdm import tqdm

from chap_model import CHAP
from util.commons import input_iterator


LABEL_VOCAB = {0: "sitting", 1: "not-sitting", -1: "no-label"}


def collect_subject_windows(data_dir, subject_id, window_size):
    """Group a subject's contiguous 10s windows into fixed-size chunks.

    Returns (x, y, ts) with shapes (M, window_size, 100, 3), (M, window_size),
    (M, window_size), or (None, None, None) if no full chunk could be formed.
    Trailing windows that don't fill a chunk within a segment are dropped, to
    match the model's fixed bi_lstm_win_size.
    """
    x_chunks, y_chunks, ts_chunks = [], [], []
    for x_seg, ts_seg, y_seg in input_iterator(data_dir, subject_id, train=False):
        n = len(y_seg)
        n_full = (n // window_size) * window_size
        if n_full == 0:
            continue
        x_chunks.append(x_seg[:n_full].reshape(-1, window_size, 100, 3))
        y_chunks.append(y_seg[:n_full].reshape(-1, window_size))
        ts_chunks.append(ts_seg[:n_full].reshape(-1, window_size))
    if not x_chunks:
        return None, None, None
    return (
        np.concatenate(x_chunks, axis=0).astype(np.float32),
        np.concatenate(y_chunks, axis=0).astype(np.int32),
        np.concatenate(ts_chunks, axis=0).astype(np.float64),
    )


def build_model(args):
    if args.model == 'CHAP':
        return CHAP(2, args.window_size, 2)
    raise ValueError(
        f"Unsupported --model {args.model!r}. This script currently supports "
        "'CHAP' (the architecture used by all SUBMIT_RESULT checkpoints)."
    )


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    msg = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint {ckpt_path}: {msg}")


@torch.no_grad()
def predict_chunks(model, x, args, device):
    """Run inference on (M, window_size, 100, 3). Returns (M*window_size,) int preds."""
    model.eval()
    preds = []
    for start in range(0, x.shape[0], args.batch_size):
        batch = torch.from_numpy(x[start:start + args.batch_size]).to(device, non_blocking=True)
        # CHAP expects (BS*window_size, 1, 100, 3); see chap_model.CHAP.forward.
        batch = rearrange(batch, 'b w l c -> (b w) 1 l c')
        logits = model(batch).view(-1)  # (BS*window_size,)
        preds.append(torch.round(torch.sigmoid(logits)).cpu().int().numpy())
    return np.concatenate(preds, axis=0)


def write_subject_csv(out_path, y_arr, ts_arr, preds):
    M, W = y_arr.shape
    has_labels = bool((y_arr != -1).any())
    timestamps = ts_arr.flatten()
    readable = [datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S") for t in timestamps]
    segment = np.repeat(np.arange(M), W)
    pred_str = [LABEL_VOCAB[int(p)] for p in preds]

    cols = {'segment': segment, 'timestamp': readable, 'prediction': pred_str}
    if has_labels:
        cols['label'] = [LABEL_VOCAB[int(l)] for l in y_arr.flatten()]
    pd.DataFrame(cols).to_csv(out_path, index=False)


def discover_subjects(data_dir, explicit):
    if explicit:
        return [s.strip() for s in explicit.split(',') if s.strip()]
    return sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')
    )


def get_args_parser():
    p = argparse.ArgumentParser('make_predictions',
        description='Run a CHAP wrist/hip checkpoint on CHAP-preprocessed daily HDF5 data.')
    p.add_argument('--data_dir', required=True,
                   help='CHAP-preprocessed daily HDF5 dir: <data_dir>/<subject_id>/<YYYY-MM-DD>.h5')
    p.add_argument('--checkpoint', required=True, help='Path to model .pth')
    p.add_argument('--prediction_dir', required=True,
                   help='Output dir for per-subject CSVs')
    p.add_argument('--subjects', default=None,
                   help='Comma-separated subject IDs (default: all subdirs of --data_dir)')
    p.add_argument('--model', default='CHAP', help='Model architecture (default: CHAP)')
    p.add_argument('--window_size', type=int, default=42,
                   help='10s windows per inference chunk; must match training (default: 42)')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    p.add_argument('--seed', type=int, default=0)
    return p


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.prediction_dir, exist_ok=True)
    device = torch.device(args.device)

    model = build_model(args).to(device)
    load_checkpoint(model, args.checkpoint)

    subjects = discover_subjects(args.data_dir, args.subjects)
    print(f"Found {len(subjects)} subject(s) under {args.data_dir}")

    for subject_id in tqdm(subjects, desc='Subjects'):
        x, y, ts = collect_subject_windows(args.data_dir, subject_id, args.window_size)
        if x is None:
            print(f"[skip] {subject_id}: no full chunk of {args.window_size} windows")
            continue
        preds = predict_chunks(model, x, args, device)
        out_path = os.path.join(args.prediction_dir, f"{subject_id}.csv")
        write_subject_csv(out_path, y, ts, preds)
        print(f"  {subject_id}: {x.shape[0]} chunks ({x.shape[0] * args.window_size} windows) -> {out_path}")


if __name__ == '__main__':
    main(get_args_parser().parse_args())
