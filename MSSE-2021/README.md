# Table of Contents
- [Table of Contents](#table-of-contents)
  - [Pre-Requisites](#pre-requisites)
  - [Data](#data)
  - [Pre-Processing Data](#pre-processing-data)
  - [Generating Predictions](#generating-predictions)
   
## Pre-Requisites
You must be running on Python 3 with the following python packages installed. We also recommend using a machine that has GPU support.

    pip install "tensorflow-gpu>=1.13.0,<2.0" # for cpu use "tensorflow>=1.13.0,<2.0"
    pip install pandas
    pip install numpy
    pip install scipy
    pip install h5py

## Data
- **Accelerometer Data**: We assume the input data is obtained from ActiGraph GT3X device and converted into single .csv files. The files should be named as **<subject_id>.csv** and files for all subjects should be put in the same directory. First few lines of a sample csv file are as follows:
    ~~~
    ------------ Data File Created By ActiGraph GT3X+ ActiLife v6.13.3 Firmware v3.2.1 date format M/d/yyyy at 30 Hz  Filter Normal -----------
    Serial Number: NEO1F18120387
    Start Time 00:00:00
    Start Date 5/7/2014
    Epoch Period (hh:mm:ss) 00:00:00
    Download Time 10:31:05
    Download Date 5/20/2014
    Current Memory Address: 0
    Current Battery Voltage: 4.07     Mode = 12
    --------------------------------------------------
    Accelerometer X,Accelerometer Y,Accelerometer Z
    -0.182,-0.182,0.962
    -0.182,-0.176,0.959
    -0.179,-0.182,0.959
    -0.179,-0.182,0.959
    ~~~

- **Valid Days File**: A .csv file indicating which dates are valid (concurrent wear days) for each subject. Each row is subject id, date pair. The header should be of the from `ID,Date.valid.day`. A sample valid days file is shown below.

    ~~~
    ID,Date.valid.day
    156976,1/19/2018
    156976,1/20/2018
    156976,1/21/2018
    156976,1/22/2018
    156976,1/23/2018
    156976,1/24/2018
    156976,1/25/2018
    ~~~

## Pre-Processing Data
First, you need to create pre-processed data from the source data. To do this invoke the `pre_process_data.py` script as follows:

    python pre_process_data.py --gt3x-dir <gt3x_data_dir> --valid-days-file <valid_days_file> --pre-processed-dir <output_dir>

Complete usage details of this script are as follows:

    usage: pre_process_data.py [-h] --gt3x-dir GT3X_DIR --valid-days-file
                           VALID_DAYS_FILE --pre-processed-dir
                           PRE_PROCESSED_DIR
                           [--sleep-logs-file SLEEP_LOGS_FILE]
                           [--non-wear-times-file NON_WEAR_TIMES_FILE]
                           [--n-start-id N_START_ID] [--n-end-id N_END_ID]
                           [--expression-after-id EXPRESSION_AFTER_ID]
                           [--window-size WINDOW_SIZE]
                           [--gt3x-frequency GT3X_FREQUENCY]
                           [--down-sample-frequency DOWN_SAMPLE_FREQUENCY]
                           [--silent]

    Argument parser for preprocessing the input data.

    required arguments:
    --gt3x-dir GT3X_DIR   GT3X data directory
    --valid-days-file VALID_DAYS_FILE
                            Path to the valid days file
    --pre-processed-dir PRE_PROCESSED_DIR
                            Pre-processed data directory

    optional arguments:
    -h, --help            show this help message and exit
    --sleep-logs-file SLEEP_LOGS_FILE
                            Path to the sleep logs file
    --non-wear-times-file NON_WEAR_TIMES_FILE
                            Path to non wear times file
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
                            made
    --gt3x-frequency GT3X_FREQUENCY
                            GT3X device frequency in Hz
    --down-sample-frequency DOWN_SAMPLE_FREQUENCY
                            Downsample frequency in Hz for GT3X data
    --silent              Whether to hide info messages


## Generating Predictions
You can use the released pre-trained models to generate predictions using your own data. To do so invoke the `make_predictions.py` as follows:

    python make_predictions.py --pre-processed-dir <pre-processed-dir> --predictions-dir <predictions-dir>

