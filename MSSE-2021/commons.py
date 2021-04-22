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
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta

def input_iterator(data_root, subject_id, train=False):
    fnames = [name.split('.')[0] for name in os.listdir(os.path.join(data_root, subject_id)) if not name.startswith('.')]
    fnames.sort()
    for i in range(len(fnames) - 1):
        assert datetime.strptime(fnames[i+1], "%Y-%m-%d").date() - datetime.strptime(fnames[i], "%Y-%m-%d").date() == timedelta(days=1)
    
    for fname in fnames:
        h5f = h5py.File(os.path.join(data_root, subject_id,  '{}.h5'.format(fname)), 'r')
        timestamps = h5f.get('time')[:]
        data = h5f.get('data')[:]
        sleeping = h5f.get('sleeping')[:]
        non_wear = h5f.get('non_wear')[:]
        label = h5f.get('label')[:]

        data_batch = []
        timestamps_batch = []
        label_batch = []
        for d, t, s, nw, l in zip(data, timestamps, sleeping, non_wear, label):
            if train and l == -1:
                raise Exception('Missing ground truth label information in pre-processed data')
            if s == 1 or nw == 1:
                if len(timestamps_batch) > 0:
                    yield np.array(data_batch), np.array(timestamps_batch), np.array(label_batch)
                data_batch = []
                timestamps_batch = []
                label_batch = []
                continue

            data_batch.append(d)
            timestamps_batch.append(t)
            label_batch.append(l)
    
        if len(timestamps_batch) > 0:
            yield np.array(data_batch), np.array(timestamps_batch), np.array(label_batch)

        h5f.close()


def cnn_bi_lstm_model(x, amp_factor, bil_lstm_win_size, num_classes):
    logits = cnn_model(x, amp_factor=amp_factor)
    logits = tf.reshape(logits, [-1, bil_lstm_win_size, 256*amp_factor])

    forward_cell = tf.nn.rnn_cell.LSTMCell(128)
    backward_cell = tf.nn.rnn_cell.LSTMCell(128)
    encoder_outputs,_ = tf.nn.bidirectional_dynamic_rnn(
            forward_cell,
            backward_cell,
            logits,
            dtype=tf.float32
        )
    encoder_outputs = tf.concat(encoder_outputs, axis=2)
    logits = tf.reshape(tf.layers.dense(encoder_outputs, units=num_classes), [-1, bil_lstm_win_size, num_classes])
    return logits
    

def cnn_model(x, amp_factor=1):
    with tf.variable_scope('model'):
        conv1 = tf.layers.conv2d(x, filters=32*amp_factor, kernel_size=[5, 3],
                                 data_format='channels_last', padding= "same",
                                 strides=(2, 1),
                                 activation=tf.nn.relu)
        pool1 = conv1

        conv2 = tf.layers.conv2d(pool1, filters=64*amp_factor, kernel_size=[5, 1],
                                 data_format='channels_last', padding= "same",
                                 strides=(2, 1),
                                 activation=tf.nn.relu)
        pool2 = conv2

        conv3 = tf.layers.conv2d(pool2, filters=128*amp_factor, kernel_size=[5, 1],
                                 data_format='channels_last', padding= "same",
                                 strides=(2, 1),
                                 activation=tf.nn.relu)
        pool3 = conv3

        conv4 = tf.layers.conv2d(pool3, filters=256*amp_factor, kernel_size=[5, 1],
                                 data_format='channels_last', padding= "same",
                                strides=(2, 1), 
                                activation=tf.nn.relu)
        pool4 = conv4

        conv5 = tf.layers.conv2d(pool4, filters=256*amp_factor, kernel_size=[5, 1],
                                 data_format='channels_last', padding= "same",
                                strides=(2, 1), 
                                activation=tf.nn.relu)
        pool5 = conv5        
        pool5 = tf.transpose(pool5, [0, 3, 1, 2])
        size = pool5.shape[-1] * pool5.shape[-2] * pool5.shape[-3]

        logits = tf.layers.dense(tf.reshape(pool5,(-1, size)), units=256*amp_factor)
        return logits
