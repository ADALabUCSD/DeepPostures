---
layout: default
title: 0. Training Your Own Model
parent: Advanced Usages
nav_order: 1
---

# Training Your Own Model

To train your own model invoke the `train_model.py` as follows:

    python train_model.py --pre-processed-dir <pre-processed-dir> --model-checkpoint-path <checkpoint-dir>

Complete usage details of this script are as follows:

    usage: train_model.py [-h] --pre-processed-dir PRE_PROCESSED_DIR
                        [--transfer-learning-model {CHAP_ALL_ADULTS, NONE}]
                        [--learning-rate LEARNING_RATE]
                        [--num-epochs NUM_EPOCHS] [--batch-size BATCH_SIZE]
                        [--amp-factor AMP_FACTOR]
                        [--cnn-window-size CNN_WINDOW_SIZE]
                        [--bi-lstm-window-size BI_LSTM_WINDOW_SIZE]
                        [--shuffle-buffer-size SHUFFLE_BUFFER_SIZE]
                        [--training-data-fraction TRAINING_DATA_FRACTION]
                        [--validation-data-fraction VALIDATION_DATA_FRACTION]
                        [--testing-data-fraction TESTING_DATA_FRACTION]
                        [--model-checkpoint-path MODEL_CHECKPOINT_PATH]
                        [--num-classes NUM_CLASSES]
                        [--class-weights CLASS_WEIGHTS]
                        [--down-sample-frequency DOWN_SAMPLE_FREQUENCY]
                        [--silent]
                        [--output-file OUTPUT_FILE]
                        [--run-test]

    Argument parser for training CNN model.

    required arguments:
    --pre-processed-dir PRE_PROCESSED_DIR
                            Pre-processed data directory

    optional arguments:
    -h, --help            show this help message and exit
    --transfer-learning-model {CHAP_ALL_ADULTS, NONE}
                            Transfer learning model name (default:
                            CHAP_ALL_ADULTS)
    --learning-rate LEARNING_RATE
                            Learning rate for training the model (default: 0.0001)
    --num-epochs NUM_EPOCHS
                            Number of epochs to train the model (default: 15)
    --batch-size BATCH_SIZE
                            Training batch size (default: 16)
    --amp-factor AMP_FACTOR
                            Factor to increase the number of neurons in the CNN
                            layers (default: 2)
    --cnn-window-size CNN_WINDOW_SIZE
                            CNN window size in seconds on which the predictions to
                            be made (default: 10)
    --bi-lstm-window-size BI_LSTM_WINDOW_SIZE
                            BiLSTM window size in minutes on which the predictions
                            to be smoothed (default: 7)
    --shuffle-buffer-size SHUFFLE_BUFFER_SIZE
                            Training data shuffle buffer size in terms of number
                            of records (default: 10000)
    --training-data-fraction TRAINING_DATA_FRACTION
                            Percentage of subjects to be used for training
                            (default: 60)
    --validation-data-fraction VALIDATION_DATA_FRACTION
                            Percentage of subjects to be used for validation
                            (default: 20)
    --testing-data-fraction TESTING_DATA_FRACTION
                            Percentage of subjects to be used for testing
                            (default: 20)
    --model-checkpoint-path MODEL_CHECKPOINT_PATH
                            Path where the trained model will be saved (default:
                            ./model-checkpoint)
    --num-classes NUM_CLASSES
                            Number of classes in the training dataset (default: 2)
    --class-weights CLASS_WEIGHTS
                            Class weights for loss aggregation (default: [1.0,
                            1.0])
    --down-sample-frequency DOWN_SAMPLE_FREQUENCY
                            Downsample frequency in Hz for GT3X data (default: 10)
    --silent              Whether to hide info messages
    --output-file OUTPUT_FILE
                            Output file to log training metric
    --run-test            Run test pipeline after training

**Model Selection:** Notice that this script relies on several hyperparameters required for training the model such as learning rate, batch size, and BiLSTM window size etc. The script comes with set of default values for these parameters. However, you may need to tweak these parameters for your dataset to get the best performance.

**Transfer Learning:** Instead of start training a model from scratch, you can also start with an existing model and tune it for your dataset. This is also called transfer learning and can be used to train a high-quality model with limited amount of training data. Currently we support transfer learning using the `CHAP_ALL_ADULTS` model. In order to use transfer learning pass the `--transfer-learning-model CHAP_ALL_ADULTS` option when invoking the `train_model.py` script. Note that with transfer learning only the training hyperparameters (e.g., batch size, learning rate, num epochs etc.) can be tweaked. The architectural hyperparameters (e.g., BiLSTM window size, CNN window size) will be set to the values of the source model. When transfer learning, it is recommeded to use a low learning rate (e.g., 0.00001) to avoid overfitting.