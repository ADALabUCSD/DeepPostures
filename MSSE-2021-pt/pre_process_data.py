# Copyright 2024 Animesh Kumar. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import time
import os
import gc
import h5py
import json
import logging
import shutil
import pandas as pd
import numpy as np
from scipy.stats import mode
from datetime import datetime
from datetime import timedelta
from functools import partial
import gzip
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def write_data_to_file(pre_process_data_output_dir, subject_id, start_date, values_being_written, CNN_WINDOW_SIZE, RESOLUTION):
    # File path
    subject_data_file_path = os.path.join(
        pre_process_data_output_dir, subject_id, "{}.h5".format(start_date.strftime("%Y-%m-%d")))

    time_values = []
    data_values = []
    sleeping_values = []
    non_wear_values = []
    label_values = []
    for j in range(int(CNN_WINDOW_SIZE / RESOLUTION), len(values_being_written) + 1, int(CNN_WINDOW_SIZE / RESOLUTION)):
        temp = values_being_written[j - int(CNN_WINDOW_SIZE / RESOLUTION):j]
        time_values.append(
            time.mktime(temp[0][0].timetuple()))
        data_values.append([[x[1], x[2], x[3]] for x in temp])
        non_wear_values.append(mode([x[4] for x in temp])[0])
        sleeping_values.append(mode([x[5] for x in temp])[0])
        label_values.append(mode([x[6] for x in temp])[0])

    # flush data, free memory
    h5f_out = h5py.File(subject_data_file_path, "w")
    h5f_out.create_dataset('time', data=np.array(
        time_values), chunks=True, maxshape=(None,))
    h5f_out.create_dataset('data', data=np.array(
        data_values), chunks=True, maxshape=(None, 100, 3))
    h5f_out.create_dataset('non_wear', data=np.array(
        non_wear_values), chunks=True, maxshape=(None,))
    h5f_out.create_dataset('sleeping', data=np.array(
        sleeping_values), chunks=True, maxshape=(None,))
    h5f_out.create_dataset('label', data=np.array(
        label_values), chunks=True, maxshape=(None,))
    h5f_out.close()


def map_function(gt3x_file, concurrent_wear_dict, sleep_logs_dict, non_wear_dict, pre_process_data_output_dir, subject_id, ap_df, label_map):
    RESOLUTION = 1/float(args.down_sample_frequency)  # seconds
    GT3X_FREQUENCY = args.gt3x_frequency  # Hz
    CNN_WINDOW_SIZE = args.window_size  # seconds

    def check_non_wear(id, check_time):
        if id in concurrent_wear_dict and check_time.replace(hour=0, minute=0, second=0, microsecond=0) not in concurrent_wear_dict[id]:
            return 1

        if id in non_wear_dict:
            for non_wear_interval in non_wear_dict[id]:
                if non_wear_interval[0] <= check_time <= non_wear_interval[1]:
                    return 1
        return 0

    def check_sleeping(id, check_time):
        if id in sleep_logs_dict:
            for sleep_interval in sleep_logs_dict[id]:
                if sleep_interval[0] <= check_time <= sleep_interval[1]:
                    return 1
            return 0
        else:
            return 0

    if ap_df is not None:
        ap_df['Time'] = ap_df['Time'].map(
            lambda x: datetime.utcfromtimestamp(round((x - 25569.) * 86400 * 10) / 10.))
        event_start_times = ap_df['Time'].tolist()
        ap_df['Interval (s)'] = ap_df['Interval (s)'].map(
            lambda x: timedelta(seconds=round(x * 10) / 10.))
        event_intervals = ap_df['Interval (s)'].tolist()
        # Fix for column name inconsistency in ActivPal
        ap_df.columns = ap_df.columns.map(lambda col: 'ActivityCode' if col.startswith('ActivityCode') else col)
        # ap_df = ap_df.rename(columns={
        #                      'ActivityCode (0=sedentary, 1= standing, 2=stepping)': 'ActivityCode (0=sedentary, 1=standing, 2=stepping)'})
        event_labels = ap_df['ActivityCode'].apply(
            lambda x: label_map[str(x)]).tolist()

    def check_label(pointer, check_time):
        if ap_df is None:
            return pointer, -1

        while pointer < len(event_start_times):
            if check_time < event_start_times[pointer]:
                return pointer, -1
            if event_start_times[pointer] <= check_time <= event_start_times[pointer] + event_intervals[pointer]:
                return pointer, event_labels[pointer]

            pointer += 1
        return pointer, -1

    if not os.path.exists(os.path.join(pre_process_data_output_dir, subject_id)):
        os.makedirs(os.path.join(pre_process_data_output_dir, subject_id))

    values = []
    gt3x_lines = [gt3x_file.readline().rstrip() for _ in range(11)]
    start_time = gt3x_lines[3][11:].strip() + " " + gt3x_lines[2][11:].strip()
    current_time = datetime.strptime(
        start_time + " UTC", "%m/%d/%Y %H:%M:%S %Z")
    unflushed_start_date = current_time.date()
    pointer = 0

    while True:
        lines = []
        while len(lines) < int(GT3X_FREQUENCY * RESOLUTION):
            l = gt3x_file.readline().rstrip()
            if len(l) == 0:
                break
            lines.append(l)

        if len(lines) < int(GT3X_FREQUENCY * RESOLUTION):
            break

        acc = np.array([[float(x) for x in l.split(',')] for l in lines])
        acc = np.mean(acc, axis=0)
        pointer, label = check_label(pointer, current_time)
        values.append([current_time, acc[0], acc[1], acc[2], check_non_wear(
            subject_id, current_time), check_sleeping(subject_id, current_time), label])

        current_time = current_time + timedelta(seconds=RESOLUTION)
        # Flush all values for a single day
        if current_time.date() > unflushed_start_date:
            if len(values) >= int(CNN_WINDOW_SIZE / RESOLUTION):
                write_data_to_file(pre_process_data_output_dir, subject_id,
                                   unflushed_start_date, values, CNN_WINDOW_SIZE, RESOLUTION)
            unflushed_start_date = current_time.date()
            gc.collect()
            values = []

    # Final flush
    if len(values) >= int(CNN_WINDOW_SIZE / RESOLUTION):
        write_data_to_file(pre_process_data_output_dir, subject_id,
                           unflushed_start_date, values, CNN_WINDOW_SIZE, RESOLUTION)
        gc.collect()


def fn(
    subject_id,
    file_name,
    non_wear_dict,
    args,
    sleep_logs_dict,
    activpal_events_csv_dir_root,
    concurrent_wear_dict,
    pre_process_data_output_dir,
    gt3x_30Hz_csv_dir_root,
    gzipped,
    ext
):
    def fn_file(fin):
        if len(non_wear_dict) > 0 and subject_id not in non_wear_dict and not args.silent:
            logger.warning(
                'Did not find non-wear records for the subject {}'.format(subject_id))
        if len(sleep_logs_dict) > 0 and subject_id not in sleep_logs_dict and not args.silent:
            logger.warning(
                'Did not find sleep log records for the subject {}'.format(subject_id))

        if not args.silent:
            logger.info(
                'Starting pre-processing for the subject {}'.format(subject_id))

        if activpal_events_csv_dir_root:
            ap_df = pd.read_csv(os.path.join(
                activpal_events_csv_dir_root, '{}.csv'.format(file_name)))
        else:
            ap_df = None

        map_function(fin, concurrent_wear_dict, sleep_logs_dict, non_wear_dict,
                     pre_process_data_output_dir, subject_id, ap_df, label_map)
        if not args.silent:
            logger.info(
                'Completed pre-processing for the subject {}'.format(subject_id))

    try:
        if len(concurrent_wear_dict) > 0 and subject_id not in concurrent_wear_dict:
            logger.warning(
                'Did not find valid days records for the subject {}'.format(subject_id))

        fullpath = os.path.join(gt3x_30Hz_csv_dir_root,
                                '{}{}'.format(file_name, ext))

        if gzipped:
            with gzip.open(fullpath, mode="rt") as fin:
                fn_file(fin)
        else:
            with open(fullpath) as fin:
                fn_file(fin)

    except Exception as e:
        logger.error(
            'Failed pre-processing for the subject {}'.format(subject_id))
        logger.error(e, exc_info=True)
        output_dir_path = os.path.join(pre_process_data_output_dir, subject_id)
        if os.path.exists(output_dir_path):
            shutil.rmtree(output_dir_path)


def get_date_string(string):
    
    patterns = [
        (r'\b\d{2}/\d{2}/\d{4}\b', '%m/%d/%Y'),  # for '%m/%d/%Y'
        (r'\b\d{4}-\d{2}-\d{2}\b', '%Y-%m-%d'),  # for '%Y-%m-%d'
    ]

    for pattern, date_format in patterns:
        if re.search(pattern, string):
            return date_format
    
    return ""

def generate_pre_processed_data(gt3x_30Hz_csv_dir_root, valid_days_file, label_map, sleep_logs_file=None, non_wear_times_file=None, activpal_events_csv_dir_root=None,
                                n_start_ID=None, n_end_ID=None, expression_after_ID=None, pre_process_data_output_dir='./pre-processed', mp=None, gzipped=False):
    """
    Utility function to generate pre-processed files from input data files that can be fed to the ML models.
    :param gt3x_30Hz_csv_dir_root: Path to the directory containing 30Hz GT3X CSV data.
    :param valid_days_file: Path to the valid days file.
    :param label_map: ActivPal label transormation map.
    :param sleep_logs_file: (Optional) Path to the sleep logs file.
    :param non_wear_times_file: (Optional) Path to non wear times file.
    :param activpal_events_csv_dir_root: (Optional) Path to the directory containing ActivPal events CSV data.
    :param n_start_ID: (Optional) The index of the starting character of the ID in gt3x file names. Indexing starts with 1.
                        If specified `n_end_ID` should also be specified. I both `n_start_ID` and `expression_after_ID` is
                        specified, the latter will be ignored.
    :param n_end_ID: (Optional) The index of the ending character of the ID in gt3x file names.
    :param expression_after_ID: (Optional) String or list of strings specifying different string spliting character that should be
                                used to identify the ID from gt3x file name. The first split will be used as the file name.
    :param pre_process_data_output_dir: Path to the directory for storing pre-precessed input files.
    """
    if gzipped:
        import gzip
        ext = '.csv.gz'
    else:
        ext = '.csv'
    # Input validations
    # 1. Check the gt3x_30Hz_csv_dir_root is a directory containing csv files.
    if not os.path.isdir(gt3x_30Hz_csv_dir_root):
        raise Exception('{} is not a directory.'.format(
            gt3x_30Hz_csv_dir_root))

    for fname in os.listdir(gt3x_30Hz_csv_dir_root):
        if not fname.startswith('.') and not fname.endswith(ext):
            raise Exception('{} directory contains unsupported file formats.'.format(
                gt3x_30Hz_csv_dir_root))

    # 2. valid_days_file
    concurrent_wear_dict = {}
    if valid_days_file is None:
        logger.warning('valid days file is not provided.')
    else:
        if not os.path.isfile(valid_days_file):
            raise Exception(
                'valid days file {} does not exists.'.format(valid_days_file))

        if not valid_days_file.endswith('.csv'):
            raise Exception('{} is not a csv file.'.format(valid_days_file))

        with open(valid_days_file) as f:
            lines = f.readlines()
            header = lines[0].lower().split(",")[:2]
            header = [head.replace("\"",'') for head in header]
            if header[0].strip().lower() != "id" or header[1].strip() != "date.valid.day":
                raise Exception(
                    'valid_days_file should have two header columns (ID, Date.valid.day).')

            for line in lines[1:]:
                line = line.strip()
                if line == "":
                    continue

                splits = line.split(",")
                splits = [split.replace("\"",'') for split in splits]
                id = splits[0].strip()
                d = splits[1].strip()
                date_string = get_date_string(d)
                try:
                    d = datetime.strptime(d, date_string)
                    if id in concurrent_wear_dict:
                        concurrent_wear_dict[id].append(d)
                    else:
                        concurrent_wear_dict[id] = [d]
                except:
                    raise Exception(
                        'In {}, Date.valid.day column in should be in (%m/%d/%Y or %Y/%m/%d or %Y-%m-%d or %m-%d-%Y) format. Found: {}.'.format(valid_days_file, line))

    # 3. sleep_logs_file
    sleep_logs_dict = {}
    if sleep_logs_file is not None:
        if not os.path.isfile(sleep_logs_file):
            raise Exception('file {} does not exist.'.format(sleep_logs_file))

        if not sleep_logs_file.endswith('.csv'):
            raise Exception('{} is not a csv file.'.format(sleep_logs_file))

        with open(sleep_logs_file) as f:
            lines = f.readlines()
            header = lines[0].lower().split(",")[:5]
            header = [head.replace("\"",'') for head in header]
            # CHAP1.0 
            if len(header) == 5:
                if header[0].strip() != "id" or header[1].strip() != "date.in.bed" or header[2].strip() != "time.in.bed" \
                        or header[3].strip() != "date.out.bed" or header[4].strip() != "time.out.bed":
                    raise Exception(
                        'sleep_logs_file should have five header columns (ID, Date.In.Bed, Time.In.Bed, Date.Out.Bed, Time.Out.Bed).')

                for line in lines[1:]:
                    line = line.strip()
                    if line == "":
                        continue

                    bits = line.split(",")
                    id = bits[0].strip()
                    if id not in sleep_logs_dict:
                        sleep_logs_dict[id] = []

                    try:
                        start_time = datetime.strptime(
                            bits[1].strip() + " " + bits[2].strip(), "%m/%d/%Y %H:%M")
                    except:
                        raise Exception(
                            "In {}, date should be in %m/%d/%Y format and time should be in %H:%M format. Found: {}".format(sleep_logs_file, line))

                    try:
                        end_time = datetime.strptime(
                            bits[3].strip() + " " + bits[4], "%m/%d/%Y %H:%M")
                    except:
                        raise Exception(
                            "In {}, date should be in %m/%d/%Y format and time should be in %H:%M format. Found: {}".format(sleep_logs_file, line))

                    sleep_logs_dict[id].append((start_time, end_time))
            elif len(header) == 3: #CHAP2.0
                if header[0].strip() != "id" or header[1].strip() != "startsleep" or header[2].strip() != "endsleep":
                    raise Exception(
                        'sleep_logs_file should have three header columns (ID, startsleep, endsleep).')

                for line in lines[1:]:
                    line = line.strip()
                    if line == "":
                        continue

                    bits = line.split(",")
                    bits = [bit.replace("\"",'') for bit in bits]
                    id = bits[0].strip()
                    if id not in sleep_logs_dict:
                        sleep_logs_dict[id] = []

                    try:
                        start_time = datetime.strptime(bits[1].strip(), "%Y-%m-%d %H:%M:%S")
                    except:
                        raise Exception(
                            "In {}, date should be in %Y-%m-%d format and time should be in %H:%M:%S format. Found: {}".format(sleep_logs_file, line))

                    try:
                        end_time = datetime.strptime(bits[2].strip(), "%Y-%m-%d %H:%M:%S")
                    except:
                        raise Exception(
                            "In {}, date should be in %Y-%m-%d format and time should be in %H:%M:%S format. Found: {}".format(sleep_logs_file, line))

                    sleep_logs_dict[id].append((start_time, end_time))
            else:
                raise Exception(
                        'sleep_logs_file should have three/five header columns. Found: {}'.format(header))




    # 4. non_wear_times_file
    non_wear_dict = {}
    if non_wear_times_file is not None:
        if not os.path.isfile(non_wear_times_file):
            raise Exception(
                'file {} does not exist.'.format(non_wear_times_file))

        if not non_wear_times_file.endswith('.csv'):
            raise Exception(
                '{} is not a csv file.'.format(non_wear_times_file))

        with open(non_wear_times_file) as f:
            lines = f.readlines()
            header = lines[0].lower().split(",")[:5]
            header = [head.replace("\"",'') for head in header]
            if header[0].strip() != "id" or header[1].strip() != "date.nw.start" or header[2].strip() != "time.nw.start" \
                    or header[3].strip() != "date.nw.end" or header[4].strip() != "time.nw.end":
                raise Exception(
                    'non_wear_times_file should have five header columns (ID, Date.nw.start, Time.nw.start, Date.nw.end, Time.nw.end).')

            for line in lines[1:]:
                line = line.strip()
                if line == "":
                    continue
                bits = line.split(",")
                bits = [bit.replace("\"",'') for bit in bits]
                id = bits[0].strip()
                if id not in non_wear_dict:
                    non_wear_dict[id] = []
                
                date_string = get_date_string(bits[1].strip())

                try:
                    start_time = datetime.strptime(
                        bits[1].strip() + " " + bits[2].strip(), f"{date_string} %H:%M")
                except:
                    raise Exception(
                        "In {}, date should be in {} format and time should be in %H:%M format. Found: {}".format(date_string, non_wear_times_file, line))

                date_string = get_date_string(bits[3].strip())

                try:
                    end_time = datetime.strptime(
                        bits[3].strip() + " " + bits[4], f"{date_string} %H:%M")
                except:
                    raise Exception(
                        "In {}, date should be in {} format and time should be in %H:%M format. Found: {}".format(date_string, non_wear_times_file, line))

                non_wear_dict[id].append((start_time, end_time))

    # 5. n_start_ID
    if n_start_ID is not None:
        if not isinstance(n_start_ID, int) or n_start_ID <= 0:
            raise Exception(
                'n_start_ID should be an integer greate than or equal to 1.')

        if n_end_ID is None or not isinstance(n_start_ID, int) or n_start_ID > n_end_ID:
            raise Exception('When n_start_ID is specified n_end_ID should also be specified. n_end_ID should be'
                            ' an integer greater than n_start_ID.')
        if expression_after_ID is not None and not args.silent:
            logger.warning(
                'Both n_start_ID and expression_after_ID specified. expression_after_ID will be ignored.')
    elif expression_after_ID is not None:
        if not isinstance(expression_after_ID, str) and \
                not (isinstance(expression_after_ID, list) and all(isinstance(x, str) for x in expression_after_ID)):
            raise Exception(
                'expression_after_ID should be a string or a list of strings.')

    # 6. Creating the pre-processed data directory if it does not exists.
    if not os.path.exists(pre_process_data_output_dir):
        os.makedirs(pre_process_data_output_dir)

    gt3x_file_names = [fname.split('.')[0] for fname in os.listdir(
        gt3x_30Hz_csv_dir_root) if fname.endswith(ext)]
    if n_start_ID is not None:
        subject_ids = []
        for x in gt3x_file_names:
            x_new = x[n_start_ID-1:n_end_ID]
            if x_new == "":
                raise Exception(
                    'Slicing the gt3x file name for extracting subject id resulted in an empty string for {}'.format(x))
            subject_ids.append(x_new)
    elif expression_after_ID is not None:
        if isinstance(expression_after_ID, str):
            expression_after_ID = [expression_after_ID]

        subject_ids = []
        for x in gt3x_file_names:
            x_new = x
            for splitter in expression_after_ID:
                splits = x.split(splitter)
                if len(splits) > 1:
                    x_new = splits[0]
                    break
            if x_new == x and not args.silent:
                logger.warning(
                    'expression_after_ID based splitting resulted no change for the gt3x file name: {}.csv'.format(x))
            subject_ids.append(x_new)
    else:
        subject_ids = gt3x_file_names

    if mp is None:
        common_args = [non_wear_dict,
                       args,
                       sleep_logs_dict,
                       activpal_events_csv_dir_root,
                       concurrent_wear_dict,
                       pre_process_data_output_dir,
                       gt3x_30Hz_csv_dir_root,
                       gzipped,
                       ext]
        for subject_id, file_name in zip(subject_ids, gt3x_file_names):
            fn(subject_id, file_name, *common_args)
    else:
        import multiprocessing
        pool = multiprocessing.Pool(processes=mp)
        fn_args = list(zip(subject_ids, gt3x_file_names))
        common_args = [non_wear_dict,
                       args,
                       sleep_logs_dict,
                       activpal_events_csv_dir_root,
                       concurrent_wear_dict,
                       pre_process_data_output_dir,
                       gt3x_30Hz_csv_dir_root,
                       gzipped,
                       ext]
        fn_args = [tuple(list(x) + common_args) for x in fn_args]
        # print(fn_args)
        pool.starmap(fn, fn_args)
    logger.info("All finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Argument parser for preprocessing the input data.')
    optional_arguments = parser._action_groups.pop()
    required_arguments = parser.add_argument_group('required arguments')
    required_arguments.add_argument(
        '--gt3x-dir', help='GT3X data directory', required=True)
    required_arguments.add_argument(
        '--pre-processed-dir', help='Pre-processed data directory', required=True)

    optional_arguments.add_argument(
        '--valid-days-file', help='Path to the valid days file', required=False)
    optional_arguments.add_argument(
        '--sleep-logs-file', help='Path to the sleep logs file', required=False)
    optional_arguments.add_argument(
        '--non-wear-times-file', help='Path to non wear times file', required=False)
    optional_arguments.add_argument(
        '--activpal-dir', help='ActivPAL data directory',  default=None, required=False)
    optional_arguments.add_argument('--n-start-id', help='The index of the starting character of the ID in gt3x file names. Indexing starts with 1. \
                        If specified `n_end_ID` should also be specified. I both `n_start_ID` and `expression_after_ID` is \
                        specified, the latter will be ignored', type=int, required=False)
    optional_arguments.add_argument(
        '--n-end-id', help='The index of the ending character of the ID in gt3x file names', type=int, required=False)
    optional_arguments.add_argument('--expression-after-id', help='String or list of strings specifying different string spliting character \
         that should be used to identify the ID from gt3x file name. The first split will be used as the file name', type=int, required=False)

    optional_arguments.add_argument(
        '--window-size', help='Window size in seconds on which the predictions to be made (default: 10)', default=10, type=int, required=False)
    optional_arguments.add_argument(
        '--gt3x-frequency', help='GT3X device frequency in Hz (default: 30)', default=30, type=int, required=False)
    optional_arguments.add_argument(
        '--down-sample-frequency', help='Downsample frequency in Hz for GT3X data (default: 10)', default=10, type=int, required=False)
    optional_arguments.add_argument(
        '--activpal-label-map', help='ActivPal label vocabulary (default: {"0": 0, "1": 1, "2": 1})', default='{"0": 0, "1": 1, "2": 1}', required=False)
    optional_arguments.add_argument(
        '--silent', help='Whether to hide info messages', default=False, required=False, action='store_true')
    optional_arguments.add_argument(
        '--mp', help='Number of concurrent workers. Hint: set to the number of cores (default: %(default)s)', type=int, default=None)
    optional_arguments.add_argument(
        '--gzipped', help='Whether the raw data is gzipped or not. Hint: extension should be .csv.gz (default: %(default)s)', default=False, action='store_true')
    parser._action_groups.append(optional_arguments)
    args = parser.parse_args()

    if not os.path.exists(args.pre_processed_dir):
        os.makedirs(args.pre_processed_dir)

    label_map = json.loads(args.activpal_label_map)
    generate_pre_processed_data(args.gt3x_dir, args.valid_days_file, label_map, sleep_logs_file=args.sleep_logs_file, non_wear_times_file=args.non_wear_times_file,
                                activpal_events_csv_dir_root=args.activpal_dir,
                                n_start_ID=args.n_start_id, n_end_ID=args.n_end_id, expression_after_ID=args.expression_after_id,
                                pre_process_data_output_dir=args.pre_processed_dir, mp=args.mp, gzipped=args.gzipped)
