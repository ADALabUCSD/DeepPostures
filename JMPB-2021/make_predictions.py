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
import sys
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import argparse

sys.path.append('./')
from commons import cnn_model, data_generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser for generating model predictions.')
    optional_arguments = parser._action_groups.pop()
    required_arguments = parser.add_argument_group('required arguments')
    required_arguments.add_argument('--pre-processed-dir', help='Pre-processed data directory', required=True)
    
    optional_arguments.add_argument('--predictions-dir', help='Training batch size', default='./predictions', required=False)
    optional_arguments.add_argument('--batch-size', help='Training batch size', default=256, type=int, required=False)
    optional_arguments.add_argument('--num-classes', help='Number of classes in the training dataset', default=3, type=int, required=False)
    optional_arguments.add_argument('--window-size', help='Window size in seconds on which the predictions to be made', default=3, type=int, required=False)
    optional_arguments.add_argument('--gt3x-frequency', help='GT3X device frequency in Hz', default=30, type=int, required=False)
    optional_arguments.add_argument('--no-label', help='Whether to not output the label', default=False, required=False, action='store_true')
    optional_arguments.add_argument('--activpal-label-map', help='ActivPal label vocabulary', default='{"sitting": 0, "standing": 1, "stepping": 2}', required=False)
    optional_arguments.add_argument('--model-checkpoint-path', help='Path where the trained model will be saved', default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pre-trained-models'), required=False)
    optional_arguments.add_argument('--remove-gravity', help='Whether to remove gravity from accelerometer data', default=False, required=False, action='store_true')
    optional_arguments.add_argument('--silent', help='Whether to hide info messages', default=False, required=False, action='store_true')
    parser._action_groups.append(optional_arguments)
    args = parser.parse_args()

    if not os.path.exists(args.predictions_dir):
        os.makedirs(args.predictions_dir)

    subject_ids = [fname.split('.')[0] for fname in os.listdir(args.pre_processed_dir) if fname.endswith('.bin')]
    
    label_map = json.loads(args.activpal_label_map)
    label_map = {label_map[k]:k for k in label_map}

    in_size = args.gt3x_frequency * args.window_size
    iterator =  tf.data.Iterator.from_structure((tf.float32, tf.int32, tf.string), ((None, 1, in_size, 3), (None, 1), (None)))
    iterator_init_ops = []
    
    for subject_id in subject_ids:
        dataset = tf.data.Dataset.from_generator(lambda: data_generator(args.pre_processed_dir, [subject_id], args.gt3x_frequency, args.remove_gravity, include_time=True), output_types=(tf.float32, tf.int32, tf.string),
                output_shapes=((1, in_size, 3), (1,), ())).batch(args.batch_size)
        iterator_init_ops.append(iterator.make_initializer(dataset))
    
    x, y, t = iterator.get_next()

    training = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)
    p = tf.argmax(cnn_model(x, args.num_classes, training), axis=1)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(args.model_checkpoint_path, 'model'))

        for subject_id, init_op in zip(subject_ids, iterator_init_ops):
            if not args.silent:
                print('Generating predictions for: {}'.format(subject_id))
            sess.run(init_op)
            ts = []
            ys = []
            ps = []
            while True:
                try:
                    temp = [v.flatten().tolist() for v in sess.run([t, y, p], feed_dict={training: False})]
                    ts.extend(temp[0])
                    ys.extend(temp[1])
                    ps.extend(temp[2])
                except tf.errors.OutOfRangeError:
                    break

            label_string = "ActivPAL activity (" + ",".join(['{}={}'.format(i, label_map[i]) for i in range(len(label_map))]) + ",-1=missing ActivPAL)"
            prediction_string = "Predicted activity (" + ",".join(['{}={}'.format(i, label_map[i]) for i in range(len(label_map))]) + ")"
            df = pd.DataFrame({'Time': ts, label_string: ys, prediction_string: ps})
            
            df['Time'] = df['Time'].str.decode("utf-8")
            
            if args.no_label:
                df = df[['Time', label_string]]
            else:
                df = df[['Time', label_string, prediction_string]]

            df.to_csv(os.path.join(args.predictions_dir, '{}.csv'.format(subject_id)), index=False)
