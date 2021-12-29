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
import sys
import numpy as np

import tensorflow
if int(tensorflow.__version__.split(".")[0]) >= 2:
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf


import pandas as pd
import random
import math
import argparse

sys.path.append('./')
from commons import cnn_bi_lstm_model, input_iterator

# Setting random seeds
tf.random.set_random_seed(2019)
random.seed(2019)
np.random.seed(2019)

def get_train_ops(y, logits, learning_rate, n_classes, class_weights):
    y = tf.reshape(y, [-1])
    logits = tf.reshape(logits, [-1, n_classes])
    balanced_accuracy, update_op = tf.metrics.mean_per_class_accuracy(y, tf.argmax(logits, 1), n_classes)
    y = tf.reshape(tf.one_hot(y, depth=n_classes, axis=1), [-1, n_classes])

    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y) * tf.reduce_sum(tf.constant(class_weights, dtype=tf.float32) * y, axis=1))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

    return train_op, update_op, balanced_accuracy, loss


def window_generator(data_root, win_size_10s, subject_ids):
    x_segments = []; y_segments = []
    for subject_id in subject_ids:
        for x_seq, _, y_seq in input_iterator(data_root, subject_id, train=True):
            x_window = []; y_window = []
            for x,y in zip(x_seq, y_seq):
                x_window.append(x)
                y_window.append(y)

                if len(y_window) == win_size_10s:
                    yield np.stack(x_window, axis=0), np.stack(y_window, axis=0)
                    x_window = []; y_window = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser for training CNN model.')
    optional_arguments = parser._action_groups.pop()
    required_arguments = parser.add_argument_group('required arguments')
    required_arguments.add_argument('--pre-processed-dir', help='Pre-processed data directory', required=True)

    optional_arguments.add_argument('--transfer-learning-model', help='Transfer learning model name (default: CHAP_ALL_ADULTS)', default=None, required=False, choices=['CHAP_ALL_ADULTS'])
    optional_arguments.add_argument('--learning-rate', help='Learning rate for training the model (default: 0.0001)', default=1e-4, type=float, required=False)
    optional_arguments.add_argument('--num-epochs', help='Number of epochs to train the model (default: 15)', default=15, type=int, required=False)
    optional_arguments.add_argument('--batch-size', help='Training batch size (default: 16)', default=16, type=int, required=False)
    
    optional_arguments.add_argument('--amp-factor', help='Factor to increase the number of neurons in the CNN layers (default: 2)', default=2, type=int, required=False)
    optional_arguments.add_argument('--cnn-window-size', help='CNN window size in seconds on which the predictions to be made (default: 10)', default=10, type=int, required=False)
    optional_arguments.add_argument('--bi-lstm-window-size', help='BiLSTM window size in minutes on which the predictions to be smoothed (default: 7)', default=7, type=int, required=False)
    
    optional_arguments.add_argument('--shuffle-buffer-size', help='Training data shuffle buffer size in terms of number of records (default: 10000)', default=10000, type=int, required=False)
    optional_arguments.add_argument('--training-data-fraction', help='Percentage of subjects to be used for training (default: 60)', default=60, type=int, required=False)
    optional_arguments.add_argument('--validation-data-fraction', help='Percentage of subjects to be used for validation (default: 20)', default=20, type=int, required=False)
    optional_arguments.add_argument('--testing-data-fraction', help='Percentage of subjects to be used for testing (default: 20)', default=20, type=int, required=False)
    optional_arguments.add_argument('--model-checkpoint-path', help='Path where the trained model will be saved (default: ./model-checkpoint)', default='./model-checkpoint', required=False)
    
    optional_arguments.add_argument('--num-classes', help='Number of classes in the training dataset (default: 2)', default=2, type=int, required=False)
    optional_arguments.add_argument('--class-weights', help='Class weights for loss aggregation (default: [1.0, 1.0])', default='[1.0, 1.0]', required=False)
    optional_arguments.add_argument('--down-sample-frequency', help='Downsample frequency in Hz for GT3X data (default: 10)', default=10, type=int, required=False)
    optional_arguments.add_argument('--silent', help='Whether to hide info messages', default=False, required=False, action='store_true')
    parser._action_groups.append(optional_arguments)
    args = parser.parse_args()

    if os.path.exists(args.model_checkpoint_path):
        raise Exception('Model checkpoint: {} already exists.'.format(args.model_checkpoint_path))

    if args.transfer_learning_model:
        if args.transfer_learning_model == 'CHAP_ALL_ADULTS':
            args.amp_factor = 2
            args.cnn_window_size = 10
            args.bi_lstm_win_size = 7
        elif args.transfer_learning_model == 'CHAP_AUSDIAB':
            args.amp_factor = 4
            args.cnn_window_size = 10
            args.bi_lstm_win_size = 9
        elif args.transfer_learning_model == 'NONE':
            raise Exception('Unsupported transfer learning model: {}'.format(args.transfer_learning_model))
    
    assert (args.training_data_fraction + args.validation_data_fraction + args.testing_data_fraction) == 100, 'Train, validation,test split fractions should add up to 100%'
    
    subject_ids = [fname.split('.')[0] for fname in os.listdir(args.pre_processed_dir)]
    random.shuffle(subject_ids)

    n_train_subjects = int(math.ceil(len(subject_ids) * args.training_data_fraction / 100.))
    train_subjects = subject_ids[:n_train_subjects]
    subject_ids = subject_ids[n_train_subjects:]

    test_frac = args.testing_data_fraction / (100.0 - args.training_data_fraction) * 100
    n_test_subjects = int(math.ceil(len(subject_ids) * test_frac / 100.))
    test_subjects = subject_ids[:n_test_subjects]
    valid_subjects = subject_ids[n_test_subjects:]    

    output_shapes = ((args.bi_lstm_window_size*(60//args.cnn_window_size), args.cnn_window_size*args.down_sample_frequency, 3), (args.bi_lstm_window_size*(60//args.cnn_window_size)))
    bi_lstm_win_size = 60//args.down_sample_frequency * args.bi_lstm_window_size
    train_dataset = tf.data.Dataset.from_generator(lambda: window_generator(args.pre_processed_dir, bi_lstm_win_size, train_subjects),output_types=(tf.float32, tf.int32),
                output_shapes=output_shapes).shuffle(args.shuffle_buffer_size).batch(args.batch_size).prefetch(10)
    valid_dataset = tf.data.Dataset.from_generator(lambda: window_generator(args.pre_processed_dir, bi_lstm_win_size, valid_subjects),output_types=(tf.float32, tf.int32),
                output_shapes=output_shapes).batch(args.batch_size).prefetch(10)
    test_dataset = tf.data.Dataset.from_generator(lambda: window_generator(args.pre_processed_dir, bi_lstm_win_size, test_subjects),output_types=(tf.float32, tf.int32),
                output_shapes=output_shapes).batch(args.batch_size).prefetch(10)
    
    iterator =  tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

    train_init_op = iterator.make_initializer(train_dataset)
    valid_init_op = iterator.make_initializer(valid_dataset)
    test_init_op = iterator.make_initializer(test_dataset)
    x, y = iterator.get_next()
    
    x = tf.reshape(x, [-1, args.cnn_window_size*args.down_sample_frequency, 3, 1])
    x = tf.identity(x, name='input')
    y = tf.reshape(y, [-1, bi_lstm_win_size])

    learning_rate = tf.placeholder(tf.float32)
    logits = cnn_bi_lstm_model(x, args.amp_factor, bi_lstm_win_size, args.num_classes)
    output = tf.argmax(tf.reshape(logits, [-1, args.num_classes]), axis=1, name='output')
    prediction = tf.identity(tf.argmax(logits, axis=1), name='prediction')

    class_weights = eval(args.class_weights)    
    train_op, update_op, balanced_accuracy, loss = get_train_ops(y, logits, learning_rate, args.num_classes, class_weights)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if args.transfer_learning_model in ['CHAP_ALL_ADULTS', 'CHAP_AUSDIAB']:
            ckpt_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pre-trained-models', '{}_CKPT'.format(args.transfer_learning_model), 'model')
            # Weights for the final classification layer (dense) are ignored
            variables = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if not v.name.startswith('dense/')]
            restorer = tf.train.Saver(variables)
            restorer.restore(sess, ckpt_path)
        
        if not args.silent:
            print('Training subjects: {}'.format(train_subjects))
            print('Validation subjects: {}'.format(valid_subjects))
            print('Testing subjects: {}'.format(test_subjects))

        for epoch in range(args.num_epochs):
            for label, init_op, subjects in zip(["Train", "Validation", "Test"],
                [train_init_op, valid_init_op, test_init_op], [train_subjects, valid_subjects, test_subjects]):
                sess.run(tf.local_variables_initializer())
                sess.run(init_op)
                losses = []
                while True:
                    try:
                        if label == "Train":
                            _, _, l = sess.run([train_op, update_op, loss], feed_dict={learning_rate: args.learning_rate})
                        elif label == "Validation":
                            _, l = sess.run([update_op, loss])
                        elif label == "Test":
                            _, l = sess.run([update_op, loss])
                        losses.append(l)
                    except tf.errors.OutOfRangeError:
                        if not args.silent:
                            ba = sess.run(balanced_accuracy)
                            print("Epoch: %d, %s Loss: %f, Balanced Accuracy: %f" %(epoch, label, sum(losses), ba))
                        break

        if not os.path.exists(args.model_checkpoint_path):
            os.makedirs(args.model_checkpoint_path)

        tf.saved_model.simple_save(sess, os.path.join(args.model_checkpoint_path, 'CUSTOM_MODEL'), inputs={"input": x}, outputs={"output": output})

        if not args.silent:
            print('Model saved in path: {}'.format(args.model_checkpoint_path))   
