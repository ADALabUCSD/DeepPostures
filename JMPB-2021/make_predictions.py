import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import argparse


tf.compat.v1.disable_eager_execution()

sys.path.append('./')
from commons import cnn_model, data_generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser for generating model predictions.')
    optional_arguments = parser._action_groups.pop()
    required_arguments = parser.add_argument_group('required arguments')
    required_arguments.add_argument('--pre-processed-dir', help='Pre-processed data directory', required=True)
    
    optional_arguments.add_argument('--predictions-dir', help='Training batch size', default='./predictions', required=False)
    optional_arguments.add_argument('--batch-size', help='Training batch size', default=256, required=False)
    optional_arguments.add_argument('--num-classes', help='Number of classes in the training dataset', default=3, required=False)
    optional_arguments.add_argument('--window-size', help='Window size in seconds on which the predictions to be made', default=5, required=False)
    optional_arguments.add_argument('--gt3x-frequency', help='GT3X device frequency in Hz', default=30, required=False)
    optional_arguments.add_argument('--no-label', help='Whether to not output the label', default=False, required=False, action='store_true')
    optional_arguments.add_argument('--model-checkpoint-path', help='Path where the trained model will be saved', default='./pre-trained-model', required=False)
    optional_arguments.add_argument('--silent', help='Whether to hide info messages', default=False, required=False, action='store_true')
    parser._action_groups.append(optional_arguments)
    args = parser.parse_args()

    if not os.path.exists(args.predictions_dir):
        os.makedirs(args.predictions_dir)

    subject_ids = [fname.split('.')[0] for fname in os.listdir(args.pre_processed_dir) if fname.endswith('.bin')]
    
    in_size = args.gt3x_frequency * args.window_size
    iterator =  tf.compat.v1.data.Iterator.from_structure((tf.float32, tf.int32, tf.string), ((None, 1, in_size, 3), (None, 1), (None)))
    iterator_init_ops = []
    
    for subject_id in subject_ids:
        dataset = tf.compat.v1.data.Dataset.from_generator(lambda: data_generator(args.pre_processed_dir, [subject_id], include_time=True), output_types=(tf.float32, tf.int32, tf.string),
                output_shapes=((1, in_size, 3), (1,), ())).batch(args.batch_size)
        iterator_init_ops.append(iterator.make_initializer(dataset))
    
    x, y, t = iterator.get_next()
    p = tf.argmax(cnn_model(x, args.num_classes), axis=1)

    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, os.path.join(args.model_checkpoint_path, 'model'))

        for subject_id, init_op in zip(subject_ids, iterator_init_ops):    
            if not args.silent:
                print('Generating predictions for: {}'.format(subject_ids))
            
            sess.run(init_op)
            ts = []
            ys = []
            ps = []
            while True:
                try:
                    temp = [v.flatten().tolist() for v in sess.run([t, y, p])]
                    ts.extend(temp[0])
                    ys.extend(temp[1])
                    ps.extend(temp[2])
                except tf.errors.OutOfRangeError:
                    break

            df = pd.DataFrame({'Time': ts, 'Label': ys, 'Prediction': ps})
            df['Time'] = df['Time'].str.decode("utf-8")
            
            if args.no_label:
                df = df[['Time', 'Prediction']]
            else:
                df = df[['Time', 'Label', 'Prediction']]
                

            df.to_csv(os.path.join(args.predictions_dir, '{}.csv'.format(subject_id)), index=False)
    
        
    