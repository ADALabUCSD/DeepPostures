import os
import numpy as np
import tensorflow as tf
import pandas as pd


def remove_gravity(acc, gt3x_frequency):
    acc = np.array(acc[0])
    alpha = 0.9
    temp = np.zeros(acc.shape)
    
    temp[0,:] = (1-alpha) * acc[0,:]

    for n in range(1, acc.shape[0]):
        temp[n,:] = alpha * temp[n-1,:] + (1-alpha) * acc[n,:]

    temp = temp[gt3x_frequency:,:] # ignore start
    gravity = np.mean(temp, axis=0)

    acc = acc - gravity

    return np.expand_dims(acc, axis=0)


def data_generator(pre_processed_dir, subjects, gt3x_frequency, no_remove_gravity, include_time=False):
    for i in subjects:
        temp = pd.read_pickle(os.path.join(pre_processed_dir, str(i)+".bin"))
        acc = temp[["Accelerometer"]].values.tolist()
        if not no_remove_gravity:
            acc = [remove_gravity(x, gt3x_frequency) for x in acc]
        timestamps = pd.to_datetime(temp.Time).dt.strftime('%Y-%m-%d %H:%M:%S').values.tolist()
        if 'Behavior' in temp.columns:
            values = temp[["Behavior"]].values
            for x, y, t in zip(acc, values, timestamps):
                if include_time:
                    yield x, y, t
                else:
                    yield x, y
        else:
            for x, t in zip(acc, timestamps):
                if include_time:
                    yield x, [-1], t
                else:
                    yield x, [-1]


def cnn_model(x, num_classes, training, keep_prob=None):
    data_format = 'channels_last'
    x = tf.transpose(x, [0, 2, 3, 1])

    conv1 = tf.compat.v1.layers.conv2d(inputs=x, filters=32, kernel_size=[5, 3], data_format=data_format, padding= "valid", activation=tf.nn.relu)
    pool1 = tf.compat.v1.layers.max_pooling2d(conv1, [2, 1], 2, padding='same', data_format=data_format)

    conv2 = tf.compat.v1.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 1], data_format=data_format, padding= "same", activation=tf.nn.relu)
    pool2 = tf.compat.v1.layers.max_pooling2d(conv2, [2, 1], 2, padding='same', data_format=data_format)

    conv3 = tf.compat.v1.layers.conv2d(inputs=pool2, filters=128, kernel_size=[5, 1], data_format=data_format, padding= "same", activation=tf.nn.relu)
    pool3 = tf.compat.v1.layers.max_pooling2d(conv3, [2, 1], 2, padding='same', data_format=data_format)

    conv4 = tf.compat.v1.layers.conv2d(inputs=pool3, filters=256, kernel_size=[5, 1], data_format=data_format, padding= "same", activation=tf.nn.relu)
    if keep_prob is not None:
        conv4 = tf.compat.v1.layers.dropout(conv4, rate=keep_prob, training=training)
    pool4 = tf.compat.v1.layers.max_pooling2d(conv4, [2, 1], 2, padding='same', data_format=data_format)

    conv5 = tf.compat.v1.layers.conv2d(inputs=pool4, filters=512, kernel_size=[5, 1], data_format=data_format, padding= "same", activation=tf.nn.relu)
    if keep_prob is not None:
        conv5 = tf.compat.v1.layers.dropout(conv5, rate=keep_prob, training=training)
    pool5 = tf.compat.v1.layers.max_pooling2d(conv5, [2, 1], 2, padding='same', data_format=data_format)

    num_features = np.prod(pool5.get_shape().as_list()[1:])

    logits = tf.compat.v1.layers.dense(inputs=tf.reshape(pool5,(-1, num_features)), units=num_classes)

    return logits