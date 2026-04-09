import os
import pickle
import h5py
import numpy as np
from tqdm import tqdm
from commons import input_iterator
import pandas as pd

def save_samples_from_iter(preprocessed_dir,
                           out_dir,
                           subject_ids,
                           window_size=42,
                           flush_threshold=1000):
    """
    Stream fixed-length windows from per-subject HDF5 files into one HDF5
    without overloading memory. We buffer up to `flush_threshold` windows in RAM,
    then write them at once.

    Outputs in out_dir/10s_train.h5:
      x           float32, shape (N_time, 100, 3)
      y           int32,   shape (N_time,)
      timestamp   float64, shape (N_time,)
      subject_id  utf-8 str, shape (N_time,)
    """
    #os.makedirs(out_dir, exist_ok=True)
    out_h5_path = out_dir#os.path.join(out_dir, '10s_val.h5')

    x_buf, y_buf, ts_buf, subj_buf = [], [], [], []
    first_write = True

    with h5py.File(out_h5_path, 'w') as f_out:
        for subject_id in tqdm(subject_ids, desc='Subjects'):
            subject_dir = os.path.join(preprocessed_dir, subject_id)
            if not os.path.isdir(subject_dir):
                continue

            for x_seq, ts_seq, y_seq in input_iterator(preprocessed_dir,
                                                       subject_id,
                                                       train=True):
                x_win, y_win, ts_win = [], [], []
                for x, ts, y in zip(x_seq, ts_seq, y_seq):
                    x_win.append(x)
                    y_win.append(y)
                    ts_win.append(ts)

                    if len(y_win) == window_size:
                        # buffer one window of shape (window_size, 100, 3)
                        x_buf.append(np.stack(x_win, axis=0).astype(np.float32))
                        y_buf.append(np.array(y_win, dtype=np.int32))
                        ts_buf.append(np.array(ts_win, dtype=np.float64))
                        subj_buf.append(subject_id)

                        x_win.clear()
                        y_win.clear()
                        ts_win.clear()

                    if len(y_buf) >= flush_threshold:
                        _flush_to_h5(f_out, x_buf, y_buf, ts_buf, subj_buf, first_write)
                        first_write = False
                        x_buf.clear(); y_buf.clear(); ts_buf.clear(); subj_buf.clear()

        # final flush
        if y_buf:
            _flush_to_h5(f_out, x_buf, y_buf, ts_buf, subj_buf, first_write)


def _flush_to_h5(f_out, x_list, y_list, ts_list, subj_list, first_write):
    """
    Write data as (BS, window_size, 100, 3) without flattening.
    """
    x_arr = np.stack(x_list, axis=0)   # (BS, window_size, 100, 3)
    y_arr = np.stack(y_list, axis=0)   # (BS, window_size)
    ts_arr = np.stack(ts_list, axis=0) # (BS, window_size)
    subj_arr = np.array(subj_list, dtype=h5py.string_dtype(encoding='utf-8'))  # (BS,)
    std_arr = np.std(x_arr, axis=2) 
    std_arr = np.mean(std_arr, axis=2)

    if first_write:
        f_out.create_dataset(
            'x',
            data=x_arr,
            maxshape=(None,) + x_arr.shape[1:],
            chunks=(min(100, x_arr.shape[0]),) + x_arr.shape[1:],
            compression='gzip'
        )
        f_out.create_dataset(
            'y',
            data=y_arr,
            maxshape=(None,) + y_arr.shape[1:],
            chunks=(min(100, y_arr.shape[0]),) + y_arr.shape[1:],
            compression='gzip'
        )
        f_out.create_dataset(
            'timestamp',
            data=ts_arr,
            maxshape=(None,) + ts_arr.shape[1:],
            chunks=(min(100, ts_arr.shape[0]),) + ts_arr.shape[1:],
            compression='gzip'
        )
        f_out.create_dataset(
            'subject_id',
            data=subj_arr,
            maxshape=(None,),
            chunks=(min(100, subj_arr.shape[0]),),
            dtype=h5py.string_dtype(encoding='utf-8'),
            compression='gzip'
        )
        f_out.create_dataset(
            'std',
            data=std_arr,
            maxshape=(None,) + std_arr.shape[1:],
            chunks=(min(100, std_arr.shape[0]),) + std_arr.shape[1:],
            compression='gzip'
        )
    else:
        for name, arr in zip(['x', 'y', 'timestamp', 'subject_id', 'std'],
                             [x_arr, y_arr, ts_arr, subj_arr, std_arr]):
            ds = f_out[name]
            old = ds.shape[0]
            new = old + arr.shape[0]
            ds.resize((new,) + ds.shape[1:])
            ds[old:new] = arr



if __name__ == "__main__":
    import argparse, shutil, tempfile

    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true',
                        help='Run demo with single subject from CHAP1_preprocess_demo/')
    parser.add_argument('--data_dir', type=str,
                        help='Pre-processed dir (required for full run)')
    parser.add_argument('--split_csv', type=str,
                        help='CSV with subject_id, split columns (required for full run)')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory for HDF5 files')
    parser.add_argument('--window_size', type=int, default=42)
    parser.add_argument('--flush_threshold', type=int, default=1000)
    args = parser.parse_args()

    if args.demo:
        # ── Demo: single subject, bundled sample data ────────────────
        # Usage: python create_dataset_split.py --demo
        #
        # Input:  DEMO/CHAP1_preprocess_demo/2018-06-25.h5  (flat)
        # But input_iterator expects <dir>/<subject_id>/<date>.h5,
        # so we create a temp dir with the right layout.
        # Output: DEMO/demo_output/10s_{train,val,test_complete}.h5
        #   (uses 10s_* naming to match iWatch dataset class)
        base_dir = os.path.join(os.path.dirname(__file__), 'DEMO')
        src_dir = os.path.join(base_dir, 'CHAP1_preprocess_demo')
        demo_subject = 'demo_subject'

        tmp_root = tempfile.mkdtemp(prefix='chap2_demo_')
        subject_dir = os.path.join(tmp_root, demo_subject)
        os.makedirs(subject_dir)
        for fname in os.listdir(src_dir):
            if fname.endswith('.h5'):
                shutil.copy2(os.path.join(src_dir, fname), subject_dir)

        out_dir = os.path.join(base_dir, 'demo_output')
        os.makedirs(out_dir, exist_ok=True)

        print(f"[DEMO] input: {src_dir}  ->  temp layout: {tmp_root}/{demo_subject}/")
        print(f"[DEMO] output: {out_dir}")

        for split_name in ['train', 'val', 'test_complete']:
            save_samples_from_iter(tmp_root,
                                   os.path.join(out_dir, f'10s_{split_name}.h5'),
                                   [demo_subject],
                                   window_size=args.window_size,
                                   flush_threshold=args.flush_threshold)

        shutil.rmtree(tmp_root)
        print(f"[DEMO] Done! -> {out_dir}/10s_train.h5, {out_dir}/10s_val.h5, {out_dir}/10s_test_complete.h5")

    else:
        # ── Full run ─────────────────────────────────────────────────
        # Usage: python create_dataset_split.py \
        #     --data_dir /niddk-data-central/SOL/PASOS/train/pre_processed_10hz \
        #     --split_csv /path/to/split.csv \
        #     --output_dir /niddk-data-central/SOL/PASOS/train/SOL_10hz
        assert args.data_dir and args.split_csv and args.output_dir, \
            "Full run requires --data_dir, --split_csv, --output_dir"

        split_df = pd.read_csv(args.split_csv)
        train_subjects = split_df[split_df['split'] == 'train']['subject_id'].tolist()
        val_subjects = split_df[split_df['split'] == 'validation']['subject_id'].tolist()
        test_subjects = split_df[split_df['split'] == 'test']['subject_id'].tolist()

        os.makedirs(args.output_dir, exist_ok=True)
        output_train_path = os.path.join(args.output_dir, '10s_train.h5')
        output_val_path = os.path.join(args.output_dir, '10s_val.h5')
        output_test_path = os.path.join(args.output_dir, '10s_test_complete.h5')

        print(f"Train: {len(train_subjects)}, Val: {len(val_subjects)}, Test: {len(test_subjects)}")

        save_samples_from_iter(args.data_dir, output_train_path, train_subjects,
                               window_size=args.window_size,
                               flush_threshold=args.flush_threshold)

        save_samples_from_iter(args.data_dir, output_val_path, val_subjects,
                               window_size=args.window_size,
                               flush_threshold=args.flush_threshold)

        save_samples_from_iter(args.data_dir, output_test_path, test_subjects,
                               window_size=args.window_size,
                               flush_threshold=args.flush_threshold)

        print("Done!")