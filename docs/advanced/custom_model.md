---
layout: default
title: 1. Use Custom-Trained Model
parent: Advanced Usages
nav_order: 2
---

# Use Custom-Trained Model

After training your own model you can use it to generate predictions by passing the model checkpoint path to the `make_predictions.py` script as follows:

    python make_predictions.py --pre-processed-dir <pre-processed-dir> --predictions-dir <predictions-dir> --model-checkpoint-path <checkpoint-dir>    

If you change the default tuning parameters during training (e.g., bi-lstm-window-size), you also need to set the same values for `make_predictions.py` by using the respective directives (e.g., `--bi-lstm-window-size`).

Complete usage details of `make_predictions.py` script with all overiding configuration values are as follows:

    usage: make_predictions.py [-h] --pre-processed-dir PRE_PROCESSED_DIR
                           [--model {CHAP_A,CHAP_B,CHAP_C,CHAP,CHAP_ALL_ADULTS,CHAP_CHILDREN,CHAP_AUSDIAB}]
                           [--predictions-dir PREDICTIONS_DIR] [--no-segment]
                           [--output-label]
                           [--model-checkpoint-path MODEL_CHECKPOINT_PATH]
                           [--cnn-window-size CNN_WINDOW_SIZE]
                           [--bi-lstm-window-size BI_LSTM_WINDOW_SIZE]
                           [--down-sample-frequency DOWN_SAMPLE_FREQUENCY]
                           [--gt3x-frequency GT3X_FREQUENCY]
                           [--activpal-label-map ACTIVPAL_LABEL_MAP]
                           [--silent] [--padding {drop,zero,wrap}]
                           [--batch-size BATCH_SIZE] [--amp-factor AMP_FACTOR]
                           [--num-classes NUM_CLASSES]

    Argument parser for generating model predictions.

    required arguments:
    --pre-processed-dir PRE_PROCESSED_DIR
                            Pre-processed data directory

    optional arguments:
    -h, --help            show this help message and exit
    --model {CHAP_A,CHAP_B,CHAP_C,CHAP,CHAP_ALL_ADULTS}
                            Pre-trained prediction model name (default:
                            CHAP_ALL_ADULTS)
    --predictions-dir PREDICTIONS_DIR
                            Predictions output directory (default: ./predictions)
    --no-segment          Do not output segment number
    --output-label        Whether to output the actual label
    --model-checkpoint-path MODEL_CHECKPOINT_PATH
                            Path where the custom trained model checkpoint is
                            located
    --cnn-window-size CNN_WINDOW_SIZE
                            CNN window size of the model in seconds on which the
                            predictions to be made (default: 10).
    --bi-lstm-window-size BI_LSTM_WINDOW_SIZE
                            BiLSTM window size in minutes (default: 7).
    --down-sample-frequency DOWN_SAMPLE_FREQUENCY
                            Downsample frequency in Hz for GT3X data (default:
                            10).
    --gt3x-frequency GT3X_FREQUENCY
                            GT3X device frequency in Hz (default: 30)
    --activpal-label-map ACTIVPAL_LABEL_MAP
                            ActivPal label vocabulary (default: {"sitting": 0,
                            "not-sitting": 1, "no-label": -1})
    --silent              Whether to hide info messages
    --padding {drop,zero,wrap}
                    Padding scheme for the last part of data that does not
                    fill a whole lstm window (default: drop)
    --batch-size BATCH_SIZE
                        Inference batch size (default: 16)
    --amp-factor AMP_FACTOR
                        Factor to increase the number of neurons in the CNN layers (default: 2)
    --num-classes NUM_CLASSES
                        Number of classes in the training dataset (default: 2)