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

- **Valid Days File**: A .csv file indicating which dates are valid (concurrent wear days) for all subjects. Each row is subject id, date pair. The header should be of the from `ID,Date.Valid.Day`.  Date values should be formatted in `%m/%d/%Y` format. A sample valid days file is shown below.

    ~~~
    ID,Date.Valid.Day
    156976,1/19/2018
    156976,1/20/2018
    156976,1/21/2018
    156976,1/22/2018
    156976,1/23/2018
    156976,1/24/2018
    156976,1/25/2018
    ~~~

- **(Optional) Sleep Logs File**: A .csv file indicating sleep records for all subjects. Each row is tuple of subject id, date went to bed, time went to bed, data came out of bed, and time went out of bed. The header should be of the form `ID,Date.In.Bed,Time.In.Bed,Date.Out.Bed,Time.Out.Bed`. Date values should be formatted in `%m/%d/%Y` format and time values should be formatted in `%H:%M` format. A sample sleep logs file is shown below.


    ~~~
    ID,Date.In.Bed,Time.In.Bed,Date.Out.Bed,Time.Out.Bed
    33333,4/28/2016,22:00,4/29/2016,10:00
    33333,4/29/2016,22:00,4/30/2016,9:00
    33333,4/30/2016,21:00,5/1/2016,8:00
    ~~~

- **(Optional) Non-wear Times File**: A .csv file indicating non-wear bouts for all subjects. Non-wear bouts can be obtained using CHOI method or something else. Each row is a tuple of subject id, non-wear bout start date, start time, end date, and end time. The header should be of the form `ID,Date.Nw.Start,Time.Nw.Start,Date.Nw.End,Time.Nw.End`. Date values should be formatted in `%m/%d/%Y` format and time values should be formatted in `%H:%M` format. A sample sleep logs file is shown below.
  
    ~~~
    ID,Date.Nw.Start,Time.Nw.Start,Date.Nw.End,Time.Nw.End
    33333,4/27/2016,21:30,4/28/2016,6:15
    33333,4/28/2016,20:40,4/29/2016,5:58
    33333,4/29/2016,22:00,4/30/2016,6:00
    ~~~

- **(Optional) Events Data**: Optionally, you can also provide ActivPal events data-especially if you wish to train your own models-for each subject as a single .csv file. These files should also be named in the **<subject_id>.csv** format and files for all subjects should be put in the same directory. Date values should be formatted in `%Y-%m-%d` format and time values should be formatted in `%H:%M:%S` format. The first few lines of a sample csv file are as follows:
    ~~~
    StartTime,EndTime,Behavior
    2014-05-07 09:47:23,2014-05-07 09:48:21,sitting
    2014-05-07 09:48:22,2014-05-07 09:48:26,standing
    2014-05-07 09:48:27,2014-05-07 09:49:03,stepping
    2014-05-07 09:49:04,2014-05-07 09:49:04,standing
    2014-05-07 09:49:05,2014-05-07 09:49:11,sitting
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

    usage: make_predictions.py [-h] --pre-processed-dir PRE_PROCESSED_DIR
                            [--model {CHAP_ACT_1,CHAP_ACT_2,CHAP_ACT_3,CHAP_ACT,CHAP_ACT_AUSDIAB}]
                            [--predictions-dir PREDICTIONS_DIR]
                            [--gt3x-frequency GT3X_FREQUENCY]
                            [--window-size WINDOW_SIZE]
                            [--model-lstm-window-sizes MODEL_LSTM_WINDOW_SIZES]
                            [--down-sample-frequency DOWN_SAMPLE_FREQUENCY]
                            [--activpal-label-map ACTIVPAL_LABEL_MAP]
                            [--model-checkpoint-path MODEL_CHECKPOINT_PATH]
                            [--no-segment] [--silent]

    Argument parser for generating model predictions.

    required arguments:
    --pre-processed-dir PRE_PROCESSED_DIR
                            Pre-processed data directory

    optional arguments:
    -h, --help            show this help message and exit
    --model {CHAP_ACT_1,CHAP_ACT_2,CHAP_ACT_3,CHAP_ACT,CHAP_ACT_AUSDIAB}
                            Prediction model name
    --predictions-dir PREDICTIONS_DIR
                            Training batch size
    --gt3x-frequency GT3X_FREQUENCY
                            GT3X device frequency in Hz
    --window-size WINDOW_SIZE
                            Window size in seconds on which the predictions to be
                            made
    --model-lstm-window-sizes MODEL_LSTM_WINDOW_SIZES
                            Model LSTM window sizes in minutes
    --down-sample-frequency DOWN_SAMPLE_FREQUENCY
                            Downsample frequency in Hz for GT3X data
    --activpal-label-map ACTIVPAL_LABEL_MAP
                            ActivPal label vocabulary
    --model-checkpoint-path MODEL_CHECKPOINT_PATH
                            Path where the trained model will be saved
    --no-segment          Do not output segment number
    --silent              Whether to hide info messages

We currently support several pre-trained models that can be used to generate predictions. They have been trained on different training datasets, which have different demographics. The recommended and default model is the `CHAP_ACT_AUSDIAB` model. However, users can change the pre-trained model to better match their needs using the `--model` option. Below we provide a summary of the available pre-trained models and the characteristics of the datasets that they were trained on.

| Model                                               | Training Dataset    |
|-----------------------------------------------------|---------------------|
|CHAP_ACT_AUSDIAB (default and recommended)           | ACT + AUSDIAB       |
|CHAP_ACT_1                                           | ACT                 |
|CHAP_ACT_2                                           | ACT                 |
|CHAP_ACT_3                                           | ACT                 |
|CHAP_ACT (ensemble of ACT_1, ACT_2, and ACT_3)        | ACT                 |

|Training Dataset | Description                                             |
|-----------------|---------------------------------------------------------|
|ACT              | ACT dataset is an older adult dataset collected from ...|
|AUSDIAB          |                                                         | 