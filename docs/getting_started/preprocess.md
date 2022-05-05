---
layout: default
title: 2. Preprocess Data
parent: Getting Started
---

# Preprocess Data
{: .no_toc }

First, you need to create pre-processed data from the source data. To do this invoke the `pre_process_data.py` script as follows:

    python pre_process_data.py --gt3x-dir <gt3x_data_dir> --pre-processed-dir <output_dir>

Complete usage details of this script are as follows:

    usage: pre_process_data.py [-h] --gt3x-dir GT3X_DIR --pre-processed-dir
                            PRE_PROCESSED_DIR
                            [--valid-days-file VALID_DAYS_FILE]
                            [--sleep-logs-file SLEEP_LOGS_FILE]
                            [--non-wear-times-file NON_WEAR_TIMES_FILE]
                            [--activpal-dir ACTIVPAL_DIR]
                            [--n-start-id N_START_ID] [--n-end-id N_END_ID]
                            [--expression-after-id EXPRESSION_AFTER_ID]
                            [--window-size WINDOW_SIZE]
                            [--gt3x-frequency GT3X_FREQUENCY]
                            [--down-sample-frequency DOWN_SAMPLE_FREQUENCY]
                            [--activpal-label-map ACTIVPAL_LABEL_MAP]
                            [--silent]

    Argument parser for preprocessing the input data.

    required arguments:
    --gt3x-dir GT3X_DIR   GT3X data directory
    --pre-processed-dir PRE_PROCESSED_DIR
                            Pre-processed data directory

    optional arguments:
    -h, --help            show this help message and exit
    --valid-days-file VALID_DAYS_FILE
                            Path to the valid days file
    --sleep-logs-file SLEEP_LOGS_FILE
                            Path to the sleep logs file
    --non-wear-times-file NON_WEAR_TIMES_FILE
                            Path to non wear times file
    --activpal-dir ACTIVPAL_DIR
                            ActivPAL data directory
    --n-start-id N_START_ID
                            The index of the starting character of the ID in gt3x
                            file names. Indexing starts with 1. If specified
                            `n_end_ID` should also be specified. I both
                            `n_start_ID` and `expression_after_ID` is specified,
                            the latter will be ignored
    --n-end-id N_END_ID   The index of the ending character of the ID in gt3x
                            file names
    --expression-after-id EXPRESSION_AFTER_ID
                            String or list of strings specifying different string
                            spliting character that should be used to identify the
                            ID from gt3x file name. The first split will be used
                            as the file name
    --window-size WINDOW_SIZE
                            Window size in seconds on which the predictions to be
                            made (default: 10)
    --gt3x-frequency GT3X_FREQUENCY
                            GT3X device frequency in Hz (default: 30)
    --down-sample-frequency DOWN_SAMPLE_FREQUENCY
                            Downsample frequency in Hz for GT3X data (default: 10)
    --activpal-label-map ACTIVPAL_LABEL_MAP
                            ActivPal label vocabulary (default: {"0": 0, "1": 1,
                            "2": 1})
    --silent              Whether to hide info messages
