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


import os
import pathlib
import logging
import numpy as np
import pandas as pd

from datetime import datetime
import argparse
import json
import pathlib
import sys
sys.path.append(pathlib.Path(__file__).parent.absolute())

from commons import get_subject_dataloader, input_iterator
from utils import load_model_weights
from model import CNNBiLSTMModel

# torch imports
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_predictions(pre_processed_data_dir, output_dir, model, segment, output_label, label_map, downsample_window, bi_lstm_window_sizes, cnn_window_size,
    gt3x_frequency, amp_factor, num_classes, model_ckpt_path, silent, batch_size, padding="drop"):
    """
    Function to generate the activity predictions for pre-precessed data. Predictions will be written out to the given
    output_dir. Predicted value will be one of 0: sedentary or 1: non-sedentary.
    :param pre_processed_data_dir: Path to the pre-processed data directory
    :param output_dir: Path to the output data directory where the predictions will be stored
    :param model: Which model to use. Avaialble options: 'CHAP,' 'CHAP_A', 'CHAP_B', 'CHAP_C', 'CHAP_ALL_ADULTS', and 'CHAP_CHILDREN' (default: 'CHAP_ALL_ADULTS').
    :param segment: Whether to output the segment number.
    :param output_label: Whether to output the actual label.
    :param label_map: Human readable label name map for predicted index.
    :param downsample_window: Downsample window size for GT3X data.
    :param bi_lstm_window_sizes: BiLSTM window sizes in minutes.
    :cnn_window_size: Window size of the CNN model in seconds.
    :gt3x_frequency: GT3X frequency.
    :model_ckpt_path: Path to the model checkpoints directory.
    """

    model = model.strip()
    if model not in ['CHAP', 'CHAP_A', 'CHAP_B', 'CHAP_C', 'CHAP_ALL_ADULTS', 'CHAP_CHILDREN', 'CHAP_AUSDIAB', 'CUSTOM_MODEL']:
        raise Exception('model should be one of: CHAP, CHAP_A, CHAP_B, CHAP_C, CHAP_ALL_ADULTS, CHAP_CHILDREN, CHAP_AUSDIAB or CUSTOM_MODEL')

    subject_ids = [fname.split('.')[0] for fname in os.listdir(pre_processed_data_dir) if not fname.startswith('.')]

    perform_ensemble = False
    if model == 'CHAP':
        models = ['CHAP_A', 'CHAP_B', 'CHAP_C']
        perform_ensemble = True
    else:
        models = [model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in models:
        # Load saved model
        bi_lstm_win_size = bi_lstm_window_sizes[model_name] * int(60*downsample_window)
        model = CNNBiLSTMModel(amp_factor=amp_factor, bi_lstm_win_size=bi_lstm_win_size, num_classes=num_classes)
        checkpoint_path = os.path.join(model_ckpt_path, f"{model_name}.pth")
        load_model_weights(model, checkpoint_path, weights_only=True)
        model.to(device)

        if not os.path.exists(os.path.join(output_dir, '{}'.format(model_name))):
            os.makedirs(os.path.join(output_dir, '{}'.format(model_name)))

        if not silent:
            logger.info('Processing model {}'.format(model_name))

        for subject_id in subject_ids:
            if not silent:
                logger.info('Starting prediction generation for the subject {}'.format(subject_id))
            data = list(input_iterator(pre_processed_data_dir, subject_id))
            x, timestamps, labels = [d[0].reshape(-1, int(1/downsample_window * cnn_window_size),
                                        int(gt3x_frequency*downsample_window), 1) for d in data], [d[1] for d in data], [d[2] for d in data]
            fout = open(os.path.join(output_dir, "{}".format(model_name), "{}.csv".format(subject_id)), 'w')

            if segment:
                fout.write('segment,')
            fout.write('timestamp')
            if output_label:
                fout.write(',label')
    
            fout.write(',prediction\n')

            for n in range(len(x)):
                border = x[n].shape[0] % bi_lstm_win_size
                wrapped = False
                zeroed = False
                if border != 0:
                    if padding == "drop":
                        x[n] = x[n][:-border]
                        timestamps[n] = timestamps[n][:-border]
                        labels[n] = labels[n][:-border]
                        # deficit = wanna_be - border
                        # print("Dropped: {} sec".format((42 - deficit) * 10))
                    else:
                        
                        deficit = bi_lstm_win_size - border
                        increment = int(1 / downsample_window)
                        labels_padded = np.full(deficit, -1)

                        if padding == "zero":
                            x_padded = np.zeros(
                                [deficit] + list(x[n].shape[1:]))
                            timestamps_padded = np.full(
                                deficit,
                                timestamps[n][-1]
                            ) + np.array(
                                [increment * (i + 1) for i in range(deficit)])
                            x[n] = np.vstack((x[n], x_padded))
                            timestamps[n] = np.hstack(
                                (timestamps[n], timestamps_padded))
                            zeroed = True
                            
                        if padding == "wrap":
                            x_last_p1 = x[n][:-border]
                            x_last_p2 = x[n][-bi_lstm_win_size:] 
                            x[n] = np.vstack((x_last_p1, x_last_p2))
                            wrapped = True
                        labels[n] = np.hstack((labels[n], labels_padded))

                # Reshape data to the dimensions of first layer
                subject_data = x[n].squeeze()
                reshaped_subject_data = np.split(subject_data, subject_data.shape[0]//bi_lstm_win_size)

                test_dataloader = get_subject_dataloader(
                test_subjects_data=reshaped_subject_data,
                batch_size=batch_size,
                )

                preds = []
                if test_dataloader != None:
                    with torch.no_grad():
                        for inputs in test_dataloader:
                            inputs = inputs.to(device, dtype=torch.float32)
                            inputs = inputs.view(-1, int(cnn_window_size * 1.0/downsample_window), 3, 1)
                            inputs = inputs.permute(0, 3, 1, 2)
                            outputs = model(inputs)
                            pred = torch.sigmoid(outputs)
                            pred = torch.round(pred)
                            np_preds = pred.cpu().detach().numpy().flatten().tolist()
                            preds+=np_preds
                            
                y_pred = np.array(preds)

                if padding == "wrap" and wrapped:
                    y_pred = np.hstack((y_pred[:-bi_lstm_win_size], y_pred[-border:]))
                elif padding == "zero" and zeroed:
                    y_pred = y_pred[:-deficit]

                for t, l, pred in zip(timestamps[n], labels[n], y_pred):
                    formatstr = ""
                    if segment:
                        formatstr += "{},{}"
                        values = [n, datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")]
                    else:
                        formatstr += "{}"
                        values = [datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")]

                    if output_label:
                        formatstr += ",{}"
                        values.append(label_map[int(l)])    

                    formatstr += ",{}\n"
                    values.append(label_map[int(pred)])

                    fout.write(formatstr.format(*values))

            fout.close()
            if not silent:
                logger.info('Completed prediction generation for the subject {}'.format(subject_id))

    if perform_ensemble:
        if not os.path.exists(os.path.join(output_dir, 'CHAP')):
            os.makedirs(os.path.join(output_dir, 'CHAP'))

        for subject_id in subject_ids:
            if not args.silent:
                logger.info('Starting enseble model prediction generation for the subject {}'.format(subject_id))

            df_1 = pd.read_csv(os.path.join(output_dir, "CHAP_A/{}.csv".format(subject_id)))
            df_2 = pd.read_csv(os.path.join(output_dir, "CHAP_B/{}.csv".format(subject_id)))
            df_3 = pd.read_csv(os.path.join(output_dir, "CHAP_C/{}.csv".format(subject_id)))

            modfied_dfs = []
            if segment:
                max_segment = max([df['segment'].max() for df in [df_1, df_2, df_3]])
                for seg in range(max_segment+1):
                    v_1 = df_1[df_1['segment'] == seg].sort_values('timestamp')
                    v_2 = df_2[df_2['segment'] == seg].sort_values('timestamp')
                    v_3 = df_3[df_3['segment'] == seg].sort_values('timestamp')
                
                    min_len = min(min(v_1.prediction.count(), v_2.prediction.count()), v_3.prediction.count())

                    if min_len > 0:
                        v_1 = v_1[:min_len]
                        v_2 = v_2[:min_len]
                        v_3 = v_3[:min_len]

                        v = v_3.copy()
                        v['predictions_A'] = v_1.prediction.values.tolist()
                        v['predictions_B'] = v_2.prediction.values.tolist()
                        v['predictions_C'] = v_3.prediction.values.tolist()

                        v.prediction = v[['predictions_A', 'predictions_B', 'predictions_C']].mode(axis='columns')
                        modfied_dfs.append(v)
            else:
                min_len = min(min(df_1.prediction.count(), df_2.prediction.count()), df_3.prediction.count())

                v_1 = df_1[:min_len]
                v_2 = df_2[:min_len]
                v_3 = df_3[:min_len]

                v = v_3.copy()
                v['predictions_A'] = v_1.prediction.values.tolist()
                v['predictions_B'] = v_2.prediction.values.tolist()
                v['predictions_C'] = v_3.prediction.values.tolist()

                v.prediction = (v['predictions_A'] + v['predictions_B'] + v['predictions_C']) / 3
                v.prediction = v.prediction.map(lambda x: round(x))
                modfied_dfs.append(v)


            if len(modfied_dfs) > 0:
                user_df = pd.concat(modfied_dfs)
                user_df.to_csv(os.path.join(output_dir, "CHAP/{}.csv".format(subject_id)), index=False)
            
            if not args.silent:
                logger.info('Completed enseble model prediction generation for the subject {}'.format(subject_id))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser for generating model predictions.')
    optional_arguments = parser._action_groups.pop()
    required_arguments = parser.add_argument_group('required arguments')
    required_arguments.add_argument('--pre-processed-dir', help='Pre-processed data directory', required=True)
    
    optional_arguments.add_argument('--model', help='Pre-trained prediction model name (default: CHAP_ALL_ADULTS)', default='CHAP_ALL_ADULTS',
        required=False, choices=['CHAP_A', 'CHAP_B', 'CHAP_C', 'CHAP', 'CHAP_ALL_ADULTS', 'CHAP_CHILDREN', 'CHAP_AUSDIAB'])
    optional_arguments.add_argument('--predictions-dir', help='Predictions output directory (default: ./predictions)', default='./predictions', required=False) 
    optional_arguments.add_argument('--no-segment', help='Do not output segment number', default=False, required=False, action='store_true')
    optional_arguments.add_argument('--output-label', help='Whether to output the actual label', default=False, required=False, action='store_true')

    optional_arguments.add_argument('--model-checkpoint-path', help='Path where the custom trained model checkpoint is located', default=None, required=False)
    optional_arguments.add_argument('--cnn-window-size', help='CNN window size of the model in seconds on which the predictions to be made (default: 10).', default=10, type=int, required=False)
    optional_arguments.add_argument('--bi-lstm-window-size', help='BiLSTM window size in minutes (default: 7).', default=None, required=False, type=int)
    optional_arguments.add_argument('--down-sample-frequency', help='Downsample frequency in Hz for GT3X data (default: 10).', default=10, type=int, required=False)   
    optional_arguments.add_argument('--gt3x-frequency', help='GT3X device frequency in Hz (default: 30)', default=30, type=int, required=False)
    optional_arguments.add_argument('--activpal-label-map', help='ActivPal label vocabulary (default: {"sitting": 0, "not-sitting": 1, "no-label": -1})', default='{"sitting": 0, "not-sitting": 1, "no-label": -1}', required=False)
    optional_arguments.add_argument('--silent', help='Whether to hide info messages', default=False, required=False, action='store_true')
    optional_arguments.add_argument(
    '--padding',
    type=str,
    help='Padding scheme for the last part of data that does not fill a whole lstm window (default: %(default)s)',
    default='drop',
    choices=('drop', 'zero', 'wrap')
    )
    optional_arguments.add_argument(
            "--batch-size",
            help="Inference batch size (default: 16)",
            default=16,
            type=int,
            required=False,
        )
    optional_arguments.add_argument(
            "--amp-factor",
            help="Factor to increase the number of neurons in the CNN layers (default: 2)",
            default=2,
            type=int,
            required=False,
        )
    optional_arguments.add_argument(
            "--num-classes",
            help="Number of classes in the training dataset (default: 2)",
            default=2,
            type=int,
            required=False,
        )
    parser._action_groups.append(optional_arguments)
    args = parser.parse_args()

    if not os.path.exists(args.predictions_dir):
        os.makedirs(args.predictions_dir)

    label_map = json.loads(args.activpal_label_map)
    label_map = {label_map[k]:k for k in label_map}

    bi_lstm_window_sizes = {"CHAP_A": 9, "CHAP_B": 9, "CHAP_C": 7, "CHAP_ALL_ADULTS": 7, "CHAP_CHILDREN": 3, "CHAP_AUSDIAB": 9}
    if args.bi_lstm_window_size is None:
        bi_lstm_window_sizes['CUSTOM_MODEL'] = 7
    else:
        bi_lstm_window_sizes[args.model] = args.bi_lstm_window_size

    
        bi_lstm_window_sizes['CUSTOM_MODEL'] = args.bi_lstm_window_size

    if args.model_checkpoint_path is not None:
        if not args.silent:
            print('Loading custom model from checkpoint path: {}'.format(args.model_checkpoint_path))
        args.model = 'CUSTOM_MODEL'
    else:
        args.model_checkpoint_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pre-trained-models-pt')

    generate_predictions(args.pre_processed_dir, output_dir=args.predictions_dir, model=args.model, segment=not args.no_segment, output_label=args.output_label,
        label_map=label_map, downsample_window=1.0/args.down_sample_frequency, bi_lstm_window_sizes=bi_lstm_window_sizes,
        cnn_window_size=args.cnn_window_size, gt3x_frequency=args.gt3x_frequency, amp_factor = args.amp_factor, num_classes = args.num_classes,
        model_ckpt_path=args.model_checkpoint_path, silent = args.silent, batch_size = args.batch_size, padding=args.padding,)
