import os
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
import random
import math
import argparse

sys.path.append('./')
from commons import cnn_model, data_generator

tf.compat.v1.disable_eager_execution()

# Setting random seeds
tf.random.set_seed(2019)
random.seed(2019)
np.random.seed(2019)


def remove_gravity(acc):
    Fs = 30
    
    g = np.zeros(acc.shape)
    x = 0.9
    g[0,:] = (1-x) * acc[0,:]

    for n in range(1, acc.shape[0]):
        g[n,:] = x * g[n-1,:] + (1-x) * acc[n,:]

    g = g[Fs:,:]
    gg = np.mean(g, axis=0)

    acc = acc - gg
    
    return acc


def get_train_ops(y, logits, learning_rate, n_classes):
    balanced_accuracy_op, balanced_accuracy = tf.compat.v1.metrics.mean_per_class_accuracy(y, tf.argmax(logits, 1), 3)
    y = tf.reshape(tf.one_hot(y, depth=n_classes, axis=1), [-1, n_classes])
    loss = tf.reduce_sum(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

    return train_op, balanced_accuracy, balanced_accuracy_op, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser for training CNN model.')
    optional_arguments = parser._action_groups.pop()
    required_arguments = parser.add_argument_group('required arguments')
    required_arguments.add_argument('--pre-processed-dir', help='Pre-processed data directory', required=True)
    
    optional_arguments.add_argument('--learning-rate', help='Learning rate for training the model', default=1e-4, required=False)
    optional_arguments.add_argument('--num-epochs', help='Number of epochs to train the model', default=15, required=False)
    optional_arguments.add_argument('--batch-size', help='Training batch size', default=256, required=False)
    optional_arguments.add_argument('--dropout-rate', help='Dropout rate during training', default=0.5, required=False)
    optional_arguments.add_argument('--shuffle-buffer-size', help='Training data shuffle buffer size in terms of number of records', default=10000, type=int, required=False)
    optional_arguments.add_argument('--training-data-fraction', help='Percentage of subjects to be used for training', default=60, required=False)
    optional_arguments.add_argument('--validation-data-fraction', help='Percentage of subjects to be used for validation', default=20, required=False)
    optional_arguments.add_argument('--testing-data-fraction', help='Percentage of subjects to be used for testing', default=20, required=False)
    optional_arguments.add_argument('--model-checkpoint-path', help='Path where the trained model will be saved', default='./model-checkpoint', required=False)
    optional_arguments.add_argument('--window-size', help='Window size in seconds on which the predictions to be made', default=3, required=False)
    optional_arguments.add_argument('--gt3x-frequency', help='GT3X device frequency in Hz', default=30, required=False)
    optional_arguments.add_argument('--num-classes', help='Number of classes in the training dataset', default=3, required=False)
    optional_arguments.add_argument('--no-remove-gravity', help='Whether not to remove gravity from accelerometer data', default=False, required=False, action='store_true')
    optional_arguments.add_argument('--silent', help='Whether to hide info messages', default=False, required=False, action='store_true')
    parser._action_groups.append(optional_arguments)
    args = parser.parse_args()

    
    assert tf.test.is_gpu_available(), 'GPU support is needed to train the CNN model.'
    assert (args.training_data_fraction + args.validation_data_fraction + args.testing_data_fraction) == 100, 'Train, validation,test split fractions should add up to 100%'
    
    subject_ids = [fname.split('.')[0] for fname in os.listdir(args.pre_processed_dir) if fname.endswith('.bin')]
    random.shuffle(subject_ids)

    n_train_subjects = int(math.ceil(len(subject_ids) * args.training_data_fraction / 100.))
    train_subjects = subject_ids[:n_train_subjects]
    subject_ids = subject_ids[n_train_subjects:]

    test_frac = args.testing_data_fraction / (100 - args.training_data_fraction) * 100
    n_test_subjects = int(math.ceil(len(subject_ids) * test_frac / 100.))
    test_subjects = subject_ids[:n_test_subjects]
    valid_subjects = subject_ids[n_test_subjects:]

    in_size = args.gt3x_frequency * args.window_size
    train_dataset = tf.compat.v1.data.Dataset.from_generator(lambda: data_generator(args.pre_processed_dir, train_subjects, args.gt3x_frequency, args.no_remove_gravity),
        output_types=(tf.float32, tf.int32), output_shapes=((1, in_size, 3), (1,))).shuffle(args.shuffle_buffer_size).batch(args.batch_size).prefetch(-1)
    valid_dataset = tf.compat.v1.data.Dataset.from_generator(lambda: data_generator(args.pre_processed_dir, valid_subjects, args.gt3x_frequency, args.no_remove_gravity),
        output_types=(tf.float32, tf.int32), output_shapes=((1, in_size, 3), (1,))).batch(args.batch_size).prefetch(-1)
    test_dataset = tf.compat.v1.data.Dataset.from_generator(lambda: data_generator(args.pre_processed_dir, test_subjects, args.gt3x_frequency, args.no_remove_gravity),
        output_types=(tf.float32, tf.int32), output_shapes=((1, in_size, 3), (1,))).batch(args.batch_size).prefetch(-1)

    iterator =  tf.compat.v1.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

    train_init_op = iterator.make_initializer(train_dataset)
    valid_init_op = iterator.make_initializer(valid_dataset)
    test_init_op = iterator.make_initializer(test_dataset)
    x, y = iterator.get_next(name='input')

    keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
    learning_rate = tf.compat.v1.placeholder(tf.float32)
    training = tf.compat.v1.placeholder(tf.bool)
    logits = cnn_model(x, args.num_classes, keep_prob)
    prediction = tf.identity(tf.argmax(logits, axis=1), name='prediction')
    
    train_op, balanced_accuracy, balanced_accuracy_op, loss = get_train_ops(y, logits, learning_rate, args.num_classes)
    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        if not args.silent:
            print('Training subjects: {}'.format(train_subjects))
            print('Validation subjects: {}'.format(valid_subjects))
            print('Testing subjects: {}'.format(test_subjects))

        for epoch in range(args.num_epochs):
            for label, init_op, subjects in zip(["Train", "Validation", "Test"],
                [train_init_op, valid_init_op, test_init_op], [train_subjects, valid_subjects, test_subjects]):
                sess.run(tf.compat.v1.local_variables_initializer())
                sess.run(init_op)
                losses = []
                while True:
                    try:
                        if label == "Train":
                            _, _, l = sess.run([train_op, balanced_accuracy, loss], feed_dict={keep_prob:args.dropout_rate, learning_rate: args.learning_rate, training:True})
                        elif label == "Validation":
                            _, l = sess.run([balanced_accuracy, loss], feed_dict={keep_prob:1.0, training:False})
                        elif label == "Test":
                            _, l = sess.run([balanced_accuracy, loss], feed_dict={keep_prob:1.0, training:False})
                        losses.append(l)
                    except tf.errors.OutOfRangeError:
                        if not args.silent:
                            ba = sess.run(balanced_accuracy_op)
                            print("Epoch: %d, %s Loss: %f, Balanced Accuracy: %f" %(epoch, label, sum(losses), ba))
                        break

        if not os.path.exists(args.model_checkpoint_path):
            os.makedirs(args.model_checkpoint_path)

        save_path = saver.save(sess, os.path.join(args.model_checkpoint_path, 'model'))

        if not args.silent:
            print('Model saved in path: {}'.format(args.model_checkpoint_path))
