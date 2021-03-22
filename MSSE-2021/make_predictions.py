# Copyright 2021 Supun Nakandala. All Rights Reserved.
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
import h5py
import pathlib
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import multiprocessing
import argparse
import json
import pathlib

import sys
sys.path.append(pathlib.Path(__file__).parent.absolute())
from commons import input_iterator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_predictions(pre_processed_data_dir, output_dir, model, segment, label_map, downsample_window, model_lstm_window_sizes, cnn_window_size,
    gt3x_frequency, model_ckpt_path):
    """
    Function to generate the activity predictions for pre-precessed data. Predictions will be written out to the given
    output_dir. Predicted value will be one of 0: sedentary or 1: non-sedentary.
    :param pre_processed_data_dir: Path to the pre-processed data directory
    :param output_dir: Path to the output data directory where the predictions will be stored
    :param model: Which model to use. Avaialble options: 'ensemble,' 'a', 'b', and 'c' (default: 'ensemble').
    :param segment: Whether to output the segment number.
    :param label_map: Human readable label name map for predicted index.
    :param downsample_window: Downsample window size for GT3X data.
    :param model_lstm_window_sizes: Model LSTM window sizes in minutes.
    :cnn_window_size: Window size of the CNN model in seconds.
    :gt3x_frequency: GT3X frequency.
    :model_ckpt_path: Path to the model checkpoints directory.
    """

    model = model.lower().strip()
    if model not in ['ensemble', 'a', 'b', 'c']:
        raise Exception('model should be one of: ensemble, a, b, or c')

    subject_ids = [fname.split('.')[0] for fname in os.listdir(pre_processed_data_dir)]

    # window size for each model in minutes
    model_window_sizes = model_lstm_window_sizes

    perform_ensemble = False
    if model == 'ensemble':
        models = ['a', 'b', 'c']
        perform_ensemble = True
    else:
        models = [model]

    for model in models:
        if not os.path.exists(os.path.join(output_dir, 'model_{}'.format(model))):
            os.makedirs(os.path.join(output_dir, 'model_{}'.format(model)))

        tf.reset_default_graph()
        p = max(1, multiprocessing.cpu_count()//2)
        sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=p, intra_op_parallelism_threads=p))
        tf.saved_model.loader.load(sess, ["serve"], os.path.join(model_ckpt_path, "model_{}".format(model)))

        for subject_id in subject_ids:
            data = list(input_iterator(pre_processed_data_dir, subject_id))
            x, timestamps = [d[0].reshape(-1, int(1/downsample_window * cnn_window_size),
                                          int(gt3x_frequency*downsample_window), 1) for d in data], [d[1] for d in data]

            fout = open(os.path.join(output_dir, "model_{}".format(model), "{}.csv".format(subject_id)), 'w')

            if segment:
                fout.write('segment,')
            fout.write('timestamp,prediction\n')

            for n in range(len(x)):
                border = x[n].shape[0] % (model_window_sizes[model] * int(60*downsample_window))
                if border != 0:
                    x[n] = x[n][:-border]
                    timestamps[n] = timestamps[n][:-border]

                y_pred = []
                for k in range(0, x[n].shape[0], model_window_sizes[model] * int(60*downsample_window)):
                    temp = x[n][k:k + model_window_sizes[model] * int(60*downsample_window)]
                    y_pred.append(sess.run('output:0', feed_dict={'input:0': temp}).flatten())

                y_pred = np.array(y_pred).flatten()
                for t, pred in zip(timestamps[n], y_pred):
                    formatstr = ""
                    if segment:
                        formatstr += "{},{},{}"
                        values = [n, datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S"), label_map[int(pred)]]
                    else:
                        formatstr += "{},{}"
                        values = [datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S"), label_map[int(pred)]]

                    formatstr += "\n"

                    fout.write(formatstr.format(*values))

            fout.close()

    if perform_ensemble:
        if not os.path.exists(os.path.join(output_dir, 'model_ensemble')):
            os.makedirs(os.path.join(output_dir, 'model_ensemble'))

        for subject_id in subject_ids:
            df_a = pd.read_csv(os.path.join(output_dir, "model_a/{}.csv".format(subject_id)))
            df_b = pd.read_csv(os.path.join(output_dir, "model_b/{}.csv".format(subject_id)))
            df_c = pd.read_csv(os.path.join(output_dir, "model_c/{}.csv".format(subject_id)))

            modfied_dfs = []
            if segment:
                max_segment = max([df['segment'].max() for df in [df_a, df_b, df_c]])
                for seg in range(max_segment+1):
                    v_a = df_a[df_a['segment'] == seg].sort_values('timestamp')
                    v_b = df_b[df_b['segment'] == seg].sort_values('timestamp')
                    v_c = df_c[df_c['segment'] == seg].sort_values('timestamp')
                
                    min_len = min(min(v_a.prediction.count(), v_b.prediction.count()), v_c.prediction.count())

                    if min_len > 0:
                        v_a = v_a[:min_len]
                        v_b = v_b[:min_len]
                        v_c = v_c[:min_len]

                        v = v_c.copy()
                        v['predictions_a'] = v_a.prediction.values.tolist()
                        v['predictions_b'] = v_b.prediction.values.tolist()
                        v['predictions_c'] = v_c.prediction.values.tolist()

                        v.prediction = v[['predictions_a', 'predictions_b', 'predictions_c']].mode(axis='columns')#(v['predictions_a'] + v['predictions_b'] + v['predictions_c']) / 3
                        # v.prediction = v.prediction.map(lambda x: round(x))
                        modfied_dfs.append(v)
            else:
                min_len = min(min(df_a.prediction.count(), df_b.prediction.count()), df_c.prediction.count())

                v_a = df_a[:min_len]
                v_b = df_b[:min_len]
                v_c = df_c[:min_len]

                v = v_c.copy()
                v['predictions_a'] = v_a.prediction.values.tolist()
                v['predictions_b'] = v_b.prediction.values.tolist()
                v['predictions_c'] = v_c.prediction.values.tolist()

                v.prediction = (v['predictions_a'] + v['predictions_b'] + v['predictions_c']) / 3
                v.prediction = v.prediction.map(lambda x: round(x))
                modfied_dfs.append(v)


            if len(modfied_dfs) > 0:
                user_df = pd.concat(modfied_dfs)
                user_df.to_csv(os.path.join(output_dir, "model_ensemble/{}.csv".format(subject_id)), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser for generating model predictions.')
    optional_arguments = parser._action_groups.pop()
    required_arguments = parser.add_argument_group('required arguments')
    required_arguments.add_argument('--pre-processed-dir', help='Pre-processed data directory', required=True)
    
    optional_arguments.add_argument('--model', help='Prediction model name', default='ensemble', required=False, choices=['a', 'b', 'c', 'ensemble'])
    optional_arguments.add_argument('--predictions-dir', help='Training batch size', default='./predictions', required=False)
    optional_arguments.add_argument('--gt3x-frequency', help='GT3X device frequency in Hz', default=30, type=int, required=False)
    optional_arguments.add_argument('--window-size', help='Window size in seconds on which the predictions to be made', default=10, type=int, required=False)
    optional_arguments.add_argument('--model-lstm-window-sizes', help='Model LSTM window sizes in minutes', default='{"a": 9, "b": 9, "c": 7}', required=False)
    optional_arguments.add_argument('--down-sample-frequency', help='Downsample frequency in Hz for GT3X data', default=10, type=int, required=False)
    optional_arguments.add_argument('--activpal-label-map', help='ActivPal label vocabulary', default='{"sitting": 0, "not-sitting": 1}', required=False)
    optional_arguments.add_argument('--model-checkpoint-path', help='Path where the trained model will be saved', default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pre-trained-models'), required=False)
    optional_arguments.add_argument('--no-segment', help='Do not output segment number', default=False, required=False, action='store_true')
    optional_arguments.add_argument('--silent', help='Whether to hide info messages', default=False, required=False, action='store_true')
    parser._action_groups.append(optional_arguments)
    args = parser.parse_args()

    if not os.path.exists(args.predictions_dir):
        os.makedirs(args.predictions_dir)

    label_map = json.loads(args.activpal_label_map)
    label_map = {label_map[k]:k for k in label_map}

    model_lstm_window_sizes = json.loads(args.model_lstm_window_sizes)

    generate_predictions(args.pre_processed_dir, output_dir=args.predictions_dir, model=args.model, segment=not args.no_segment,
        label_map=label_map, downsample_window=1.0/args.down_sample_frequency, model_lstm_window_sizes=model_lstm_window_sizes,
        cnn_window_size=args.window_size, gt3x_frequency=args.gt3x_frequency, model_ckpt_path=args.model_checkpoint_path)

