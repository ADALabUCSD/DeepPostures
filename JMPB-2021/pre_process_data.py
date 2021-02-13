import os
import time
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import argparse

LABEL_MAP = {'sitting': 0, 'standingStill': 1, 'walking/running': 2}
ACC_FREQUENCY = 30


def preprocess_raw_data(gt3x_dir, activpal_dir, user_id):
    if activpal_dir is not None:
        # Read activepal file
        def date_parser(x): return pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        df_ap = pd.read_csv(os.path.join(activpal_dir, str(user_id)+'.csv'),
                            parse_dates=['StartTime', 'EndTime'], date_parser=date_parser, usecols=['StartTime', 'EndTime', 'behavior'])

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
                             timedelta(seconds=i), LABEL_MAP[x['behavior']]])

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
    for i in range(0, len(df_acc), 30):
        x = np.array(df_acc.iloc[i:i+30])
        data.append([begin_time + timedelta(seconds=i//ACC_FREQUENCY), x])

    df_acc = pd.DataFrame(data)
    df_acc.columns = ['Time', 'Accelerometer']

    # Create joined table
    if df_ap is not None:
        df = pd.merge(df_acc, df_ap, on='Time')
        df['User'] = user_id
        df = df[['User', 'Segment', 'Time', 'Accelerometer', 'Behavior']]
    else:
        df['User'] = user_id
        df = df[['User', 'Time', 'Accelerometer']]

    return df


def extract_windows(original_df, window_size):
    df = []
    for (user, segment), group in original_df.groupby(["User", "Segment"]):
        group.index = group["Time"]
        group = group[~group.index.duplicated(keep='first')]
        temp1 = group["Accelerometer"].resample(
            str(window_size)+'s', base=group.iloc[0][2].second).apply(lambda x: np.vstack(x.values.tolist()))[:-1]
        temp2 = group["Behavior"].resample(str(window_size)+'s', base=group.iloc[0][2].second)\
            .apply(lambda x: x.values.tolist()[0] if len(x.values.tolist()) == 1 else -1)[:-1]
        temp = pd.concat([temp1, temp2], axis=1)

        temp["User"] = user
        temp["Segment"] = segment

        temp = temp[["User", "Segment", "Accelerometer", "Behavior"]]
        temp = temp[temp["Behavior"] >= 0]

        df.append(temp)

    return pd.concat(df)


def extract_features(gt3x_dir, activpal_dir, pre_processed_dir, user_id):
    df = preprocess_raw_data(gt3x_dir, activpal_dir, user_id)
    df = extract_windows(df, window_size=3)
    df = df[['User', 'Segment', 'Accelerometer', 'Behavior']]

    # write the joined table
    df.to_pickle(os.path.join(pre_processed_dir, str(user_id)+'.bin'))



# Example invocation
# python pre_process_data.py --gt3x-dir ./data/gt3x --pre-processed-dir ./data/pre-processed --activpal-dir ./data/activpal
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser for preprocessing the input data.')
    parser.add_argument('--gt3x-dir', help='GT3X data directory')
    parser.add_argument('--pre-processed-dir', help='Pre-processed data directory')
    parser.add_argument('--activpal-dir', help='ActivPAL data directory')
    args = parser.parse_args()

    if args.gt3x_dir is None:
        raise Exception('Please provide a value for the --gt3x-dir option!')
    if args.pre_processed_dir is None:
        raise Exception('Please provide a value for the --pre-processed-dir option!')

    for fname in os.listdir(args.gt3x_dir):
        if fname.endswith('.csv'):
            user_id = fname.split(".")[0]
            extract_features(args.gt3x_dir, args.activpal_dir, args.pre_processed_dir, user_id)
            print('Completed pre-processing data for subject: {}'.format(user_id))
