# import os
# import h5py
# import numpy as np
# from tqdm import tqdm
# from commons import input_iterator
# import pickle

# import os
# import numpy as np
# import h5py

# def merge(h_f5, d_f5, subject_ids, output_dir, tol=1e-10):
#     """
#     Iterate over subject_ids, join hip (h_f5) and wrist (d_f5) by exact timestamp,
#     merging their 3-channel windows into 6-channel windows.

#     - h_f5, d_f5: paths to HDF5 files containing datasets:
#         'x': (N, 100, 3), 'y': (N,), 'timestamp': (N,), 'subject_id': (N,)
#     - subject_ids: list of subject_id strings to process
#     - output_dir: directory where 'merged_data.h5' and 'merge_warnings.log' are created
#     - tol: unused for integer timestamps (exact match required)

#     Records any dropped windows in 'merge_warnings.log'.

#     Returns the path to the merged HDF5 file.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     out_h5_path = os.path.join(output_dir, 'merged_val_data.h5')
#     log_path = os.path.join(output_dir, 'merge_val_warnings.log')

#     # Initialize log file
#     with open(log_path, 'w') as log_f:
#         log_f.write('subject_id\ttimestamp\treason\n')

#     # Open input files and preload lightweight arrays
#     with h5py.File(h_f5, 'r') as f_hip, h5py.File(d_f5, 'r') as f_wrist:
#         hip_subj = f_hip['subject_id'][:].astype('U')  # array of bytes or str
#         wrist_subj = f_wrist['subject_id'][:].astype('U')
#         hip_ts = f_hip['timestamp'][:]
#         wrist_ts = f_wrist['timestamp'][:]
#         hip_x = f_hip['x']
#         hip_y = f_hip['y']
#         wrist_x = f_wrist['x']
#         wrist_y = f_wrist['y']

#         # Set up output HDF5 with extendable datasets
#         with h5py.File(out_h5_path, 'w') as f_out:
#             # merged x will have shape (None, 100, 6)
#             merged_shape = (0, hip_x.shape[1], hip_x.shape[2] * 2)
#             chunk_shape = (1000, hip_x.shape[1], hip_x.shape[2] * 2)
#             x_out = f_out.create_dataset(
#                 'x', shape=merged_shape, maxshape=(None, 100, 6),
#                 chunks=chunk_shape, dtype=hip_x.dtype
#             )
#             y_out = f_out.create_dataset(
#                 'y', shape=(0,), maxshape=(None,),
#                 chunks=(1000,), dtype=hip_y.dtype
#             )
#             ts_out = f_out.create_dataset(
#                 'timestamp', shape=(0,), maxshape=(None,),
#                 chunks=(1000,), dtype=hip_ts.dtype
#             )
#             subj_out = f_out.create_dataset(
#                 'subject_id', shape=(0,), maxshape=(None,),
#                 chunks=(1000,), dtype=h5py.string_dtype(encoding='utf-8')
#             )

#             total = 0
#             # Process each subject separately
#             for subject_id in tqdm(subject_ids):
#                 # Find indices for this subject

#                 hip_idx = np.where(hip_subj == subject_id)[0]
#                 wrist_idx = np.where(wrist_subj == subject_id)[0]
#                 if hip_idx.size == 0 or wrist_idx.size == 0:
#                     with open(log_path, 'a') as log_f:
#                         log_f.write(f"{subject_id}\t-\tno windows in one file\n")
#                     continue

#                 # Build quick lookup for hip timestamps
#                 hip_ts_sub = hip_ts[hip_idx]
#                 hip_map = {ts: idx for ts, idx in zip(hip_ts_sub, hip_idx)}

#                 # Iterate wrist windows and merge when timestamp matches
#                 for w_i in wrist_idx:
#                     ts = wrist_ts[w_i]
#                     hip_i = hip_map.get(ts)
#                     if hip_i is None:
#                         with open(log_path, 'a') as log_f:
#                             log_f.write(f"{subject_id}\t{ts}\tmissing counterpart\n")
#                         continue

#                     # Read data and check labels
#                     x1 = hip_x[hip_i]
#                     x2 = wrist_x[w_i]
#                     y1 = hip_y[hip_i]
#                     y2 = wrist_y[w_i]
#                     if y1 != y2:
#                         with open(log_path, 'a') as log_f:
#                             log_f.write(f"{subject_id}\t{ts}\tlabel mismatch\n")
#                         continue
#                     #print(x1.shape, x2.shape) # (100,3) (100,3)
#                     merged_x = np.concatenate([x1, x2], axis=1)

#                     # Resize and append to output
#                     new_total = total + 1
#                     x_out.resize((new_total, 100, 6))
#                     y_out.resize((new_total,))
#                     ts_out.resize((new_total,))
#                     subj_out.resize((new_total,))

#                     x_out[total] = merged_x
#                     y_out[total] = y1
#                     ts_out[total] = ts
#                     subj_out[total] = subject_id

#                     total = new_total
                    
#     print('Done merging:', total, 'windows')
#     return out_h5_path



# if __name__ == "__main__":
#     preprocessed_h = '/niddk-data-central/iWatch/pre_processed_seg/H/10s_val.h5'
#     preprocessed_w = '/niddk-data-central/iWatch/pre_processed_seg/W/10s_val.h5'
#     output_dir     = '/niddk-data-central/iWatch/pre_processed_seg/HW'

#     with open("/niddk-data-central/iWatch/support_files/iwatch_split_dict.pkl", "rb") as f:
#         split_data = pickle.load(f)
#     train_subjects = split_data["val"]

#     merge(preprocessed_h, preprocessed_w, train_subjects, output_dir)
#     print('Done!')


# # python merge_hip_wrist.py


import os
import h5py
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count, Lock, Manager
from tqdm import tqdm

def init_pool(l):
    global lock
    lock = l

def process_file_pair_with_log(args):
    h_path, w_path, hw_path, log_file_path = args

    try:
        with h5py.File(h_path, 'r') as f_h, h5py.File(w_path, 'r') as f_w:
            time_h = f_h['time'][:]
            time_w = f_w['time'][:]
            label_h = f_h['label'][:]
            label_w = f_w['label'][:]
            non_wear_h = f_h['non_wear'][:]
            non_wear_w = f_w['non_wear'][:]
            sleeping_h = f_h['sleeping'][:]
            sleeping_w = f_w['sleeping'][:]

            subject = Path(h_path).parts[-2]
            date = Path(h_path).stem

            h_map = {ts: i for i, ts in enumerate(time_h)}
            matched, unmatched, label_mismatch = [], [], 0

            for i, ts in enumerate(time_w):
                if ts in h_map:
                    i_h = h_map[ts]
                    if label_h[i_h] != label_w[i]:
                        label_mismatch += 1
                    else:
                        matched.append((i_h, i))
                else:
                    unmatched.append(i)

            if unmatched or label_mismatch:
                with lock, open(log_file_path, 'a') as log_f:
                    if unmatched:
                        log_f.write(f"{subject}\t{date}\t{len(unmatched)} unmatched\n")
                    if label_mismatch:
                        log_f.write(f"{subject}\t{date}\t{label_mismatch} label mismatch\n")

            if not matched:
                return 0

            merged_x, merged_label, merged_non_wear, merged_sleeping, merged_time = [], [], [], [], []

            for i_h, i_w in matched:
                x1 = f_h['data'][i_h]
                x2 = f_w['data'][i_w]
                merged_x.append(np.concatenate([x1, x2], axis=1))
                merged_label.append(label_h[i_h])
                merged_non_wear.append(non_wear_h[i_h] | non_wear_w[i_w])
                merged_sleeping.append(sleeping_h[i_h] | sleeping_w[i_w])
                merged_time.append(time_h[i_h])

            Path(hw_path).parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(hw_path, 'w') as f_hw:
                f_hw.create_dataset('data', data=np.stack(merged_x))
                f_hw.create_dataset('label', data=np.array(merged_label))
                f_hw.create_dataset('non_wear', data=np.array(merged_non_wear))
                f_hw.create_dataset('sleeping', data=np.array(merged_sleeping))
                f_hw.create_dataset('time', data=np.array(merged_time))

            return 1

    except Exception as e:
        with lock, open(log_file_path, 'a') as log_f:
            log_f.write(f"ERROR\t{os.path.basename(h_path)}\t{str(e)}\n")
        return 0



def merge_hip_wrist_structure_parallel_with_log(h_root, w_root, hw_root, log_file_path):
    tasks = []
    for root, _, files in os.walk(h_root):
        for file in files:
            if file.endswith(".h5"):
                h_path = os.path.join(root, file)
                relative_path = os.path.relpath(h_path, h_root)
                w_path = os.path.join(w_root, relative_path)
                hw_path = os.path.join(hw_root, relative_path)
                if os.path.exists(w_path):
                    tasks.append((h_path, w_path, hw_path, log_file_path))

    Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_file_path, 'w') as f:
        f.write("source\ttime\treason\n")  # write header only once

    merged_count = 0
    l = Lock()

    with Pool(cpu_count(), initializer=init_pool, initargs=(l,)) as pool:
        for result in tqdm(pool.imap_unordered(process_file_pair_with_log, tasks), total=len(tasks)):
            merged_count += result

    print(f"Done. Merged {merged_count} files. Log available at {log_file_path}")


# Run
merge_hip_wrist_structure_parallel_with_log(
    h_root="/niddk-data-central/iWatch/pre_processed_pt/H",
    w_root="/niddk-data-central/iWatch/pre_processed_pt/W",
    hw_root="/niddk-data-central/iWatch/pre_processed_pt/HW",
    log_file_path="log.txt"
)
