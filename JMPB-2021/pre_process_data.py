
# Copyright 2020 Supun Nakandala. All Rights Reserved.
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

import os
import time
import json
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import argparse


def filter_labels(x):
    if len(set(x)) == 1:
        return x[0]
    else:
        return -1


def preprocess_raw_data(gt3x_dir, activpal_dir, user_id, gt3x_frequency, label_map):
    if activpal_dir is not None:
        # Read activepal file
        def date_parser(x): return pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        df_ap = pd.read_csv(os.path.join(activpal_dir, str(user_id)+'.csv'),
                            parse_dates=['StartTime', 'EndTime'], date_parser=date_parser, usecols=['StartTime', 'EndTime', 'Behavior'])

        # Flatten the activepal file to 1 second resolution
        data = []
        prev_end_time = None
        segment_no = 0
        for i in range(len(df_ap)):
            x = df_ap.iloc[i]

            if not (prev_end_time is None) and (x['StartTime']-prev_end_time).total_seconds() > 1:
                segment_no += 1

            for i in range(int((x['EndTime']-x['StartTime']).total_seconds() + 1)):
                data.append([segment_no, x['StartTime'] +
                             timedelta(seconds=i), label_map[x['Behavior']]])

            prev_end_time = x['EndTime']

        df_ap = pd.DataFrame(data)
        df_ap.columns = ['Segment', 'Time', 'Behavior']
    else:
        df_ap = None

    # Find activegraph start time
    with open(os.path.join(gt3x_dir, str(user_id)+'.csv'), 'r') as fp:
        acc_start_time = ''
        count = 0
        for l in fp:
            if count == 2:
                acc_start_time = l.split(' ')[2].strip()
            elif count == 3:
                acc_start_time = l.split(' ')[2].strip() + ' ' + acc_start_time
                break
            count += 1

    # Read activegraph file
    df_acc = pd.read_csv(os.path.join(gt3x_dir, str(user_id)+'.csv'), skiprows=10)

    # Aggregate at 1 second resolution
    data = []
    begin_time = datetime.strptime(acc_start_time, '%m/%d/%Y %H:%M:%S')
    for i in range(0, len(df_acc), gt3x_frequency):
        x = np.array(df_acc.iloc[i:i+gt3x_frequency])
        data.append([begin_time + timedelta(seconds=i//gt3x_frequency), x])

    df_acc = pd.DataFrame(data)
    df_acc.columns = ['Time', 'Accelerometer']

    # Create joined table
    if df_ap is not None:
        df = pd.merge(df_acc, df_ap, on='Time')
        df['User'] = user_id
        df = df[['User', 'Segment', 'Time', 'Accelerometer', 'Behavior']]
    else:
        df = df_acc
        df['User'] = user_id
        df = df[['User', 'Time', 'Accelerometer']]

    return df


def extract_windows(original_df, window_size):
    df = []
    for (user, segment), group in original_df.groupby(["User", "Segment"]):
        group.index = group["Time"]
        group = group[~group.index.duplicated(keep='first')]
        # [:-1] becuase the last row may not necessarily have window_size seconds of data
        temp = group["Accelerometer"].resample(str(window_size)+'s', base=group.iloc[0][2].second).apply(lambda x: np.vstack(x.values.tolist()))[:-1]
        
        temp2 = group["Time"].resample(str(window_size)+'s', base=group.iloc[0][2].second).apply(lambda x: x.values.tolist()[0])
        temp = pd.concat([temp, temp2], axis=1)[:-1]

        if 'Behavior' in original_df.columns:
            temp2 = group["Behavior"].resample(str(window_size)+'s', base=group.iloc[0][2].second).apply(lambda x: filter_labels(x.values.tolist()))
            temp = pd.concat([temp, temp2], axis=1)[:-1]
            # Remove time windows with more than one label
            temp = temp[temp["Behavior"] >= 0]

        temp["User"] = user
        temp["Segment"] = segment

        if 'Behavior' in original_df.columns:
            temp = temp[["User", "Segment", "Time", "Accelerometer", "Behavior"]]
            temp = temp[temp["Behavior"] >= 0]
        else:
            temp = temp[["User", "Segment", "Time", "Accelerometer"]]

        df.append(temp)

    return pd.concat(df)


def extract_features(gt3x_dir, activpal_dir, pre_processed_dir, user_id, window_size, gt3x_frequency, label_map):
    df = preprocess_raw_data(gt3x_dir, activpal_dir, user_id, gt3x_frequency, label_map)
    if activpal_dir is None:
        df['Segment'] = 0
        df = df[['User', 'Segment', 'Time', 'Accelerometer']]
    
    # We use a window of 3
    df = extract_windows(df, window_size=window_size)

    # Write the joined table
    df.to_pickle(os.path.join(pre_processed_dir, str(user_id)+'.bin'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser for preprocessing the input data.')
    optional_arguments = parser._action_groups.pop()
    required_arguments = parser.add_argument_group('required arguments')
    required_arguments.add_argument('--gt3x-dir', help='GT3X data directory', required=True)
    required_arguments.add_argument('--pre-processed-dir', help='Pre-processed data directory', required=True)
    
    optional_arguments.add_argument('--activpal-dir', help='ActivPAL data directory', default=None, required=False)
    optional_arguments.add_argument('--window-size', help='Window size in seconds on which the predictions to be made', default=3, type=int, required=False)
    optional_arguments.add_argument('--gt3x-frequency', help='GT3X device frequency in Hz', default=30, type=int, required=False)
    optional_arguments.add_argument('--activpal-label-map', help='ActivPal label vocabulary', default='{"sitting": 0, "standingStill": 1, "walking/running": 2}', required=False)
    optional_arguments.add_argument('--silent', help='Whether to hide info messages', default=False, required=False, action='store_true')
    parser._action_groups.append(optional_arguments)
    args = parser.parse_args()

    if not os.path.exists(args.pre_processed_dir):
        os.makedirs(args.pre_processed_dir)

    label_map = json.loads(args.activpal_label_map)

    for fname in os.listdir(args.gt3x_dir):
        if fname.endswith('.csv'):
            user_id = fname.split(".")[0]
            extract_features(args.gt3x_dir, args.activpal_dir, args.pre_processed_dir, user_id, args.window_size, args.gt3x_frequency, label_map)
            if not args.silent:
                print('Completed pre-processing data for subject: {}'.format(user_id))
