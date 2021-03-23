# Table of Contents
- [Table of Contents](#table-of-contents)
  - [Pre-Requisites](#pre-requisites)
  - [Data](#data)
  - [Pre-Processing Data](#pre-processing-data)
  - [Generating Predictions](#generating-predictions)
  - [Training Your Own Model](#training-your-own-model)
   
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

- **(Optional) Events Data**: Optionally, you can also provide ActivPal events data-especially if you wish to train your own models-for each subject as a single .csv file. These files should also be named in the **<subject_id>.csv** format and files for all subjects should be put in the same directory. The first few lines of a sample csv file are as follows:
    ~~~
    "Time","DataCount (samples)","Interval (s)","ActivityCode (0=sedentary, 1= standing, 2=stepping)","CumulativeStepCount","Activity Score (MET.h)","Abs(sumDiff)"
    42633.4165162037,0,1205.6,0,0,.4186111,171
    42633.4304699074,12056,3.5,1,0,1.361111E-03,405
    42633.4305104167,12091,1.4,2,1,1.266667E-03,495
    42633.4305266204,12105,.9,2,2,1.072222E-03,340
    42633.430537037,12114,1,2,3,1.111111E-03,305
    42633.4305486111,12124,1,2,4,1.111111E-03,338
    42633.4305601852,12134,1,2,5,1.111111E-03,297
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
    
    python make_predictions.py --pre-processed-dir <pre_processed_data_dir>

Complete usage details of this script are as follows:

    usage: make_predictions.py [-h] --pre-processed-dir PRE_PROCESSED_DIR
                            [--model {CHAP_ACT_1,CHAP_ACT_2,CHAP_ACT_3,CHAP_ACT,CHAP_ACT_AUSDIAB}]
                            [--predictions-dir PREDICTIONS_DIR] [--no-segment]
                            [--output-label]
                            [--model-checkpoint-path MODEL_CHECKPOINT_PATH]
                            [--cnn-window-size CNN_WINDOW_SIZE]
                            [--bi-lstm-window-size BI_LSTM_WINDOW_SIZE]
                            [--down-sample-frequency DOWN_SAMPLE_FREQUENCY]
                            [--gt3x-frequency GT3X_FREQUENCY]
                            [--activpal-label-map ACTIVPAL_LABEL_MAP]
                            [--silent]

    Argument parser for generating model predictions.

    required arguments:
    --pre-processed-dir PRE_PROCESSED_DIR
                            Pre-processed data directory

    optional arguments:
    -h, --help            show this help message and exit
    --model {CHAP_ACT_1,CHAP_ACT_2,CHAP_ACT_3,CHAP_ACT,CHAP_ACT_AUSDIAB}
                            Pre-trained prediction model name (default:
                            CHAP_ACT_AUSDIAB)
    --predictions-dir PREDICTIONS_DIR
                            Training batch size
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
                            ActivPal label vocabulary
    --silent              Whether to hide info messages

We currently support several pre-trained models that can be used to generate predictions. They have been trained on different training datasets, which have different demographics. The recommended and default model is the `CHAP_ACT_AUSDIAB` model. However, users can change the pre-trained model to better match their needs using the `--model` option. Below we provide a summary of the available pre-trained models and the characteristics of the datasets that they were trained on.

| Model                                               | Training Dataset    |
|-----------------------------------------------------|---------------------|
|CHAP_ACT_AUSDIAB (default and recommended)           | ACT + AUSDIAB       |
|CHAP_ACT_1                                           | ACT                 |
|CHAP_ACT_2                                           | ACT                 |
|CHAP_ACT_3                                           | ACT                 |
|CHAP_ACT (ensemble of ACT_1, ACT_2, and ACT_3)       | ACT                 |

|Training Dataset | Description                                             |
|-----------------|---------------------------------------------------------|
|ACT              | ACT dataset is an older adult dataset collected from ...|
|AUSDIAB          |                                                         | 

## Training Your Own Model
To train your own model invoke the `train_model.py` as follows:

    python --pre-processed-dir <pre-processed-dir> --model-checkpoint-path <checkpoint-dir>

Complete usage details of this script are as follows:

    usage: train_model.py [-h] --pre-processed-dir PRE_PROCESSED_DIR
                        [--warm-start-model {CHAP_ACT_AUSDIAB}]
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

    Argument parser for training CNN model.

    required arguments:
    --pre-processed-dir PRE_PROCESSED_DIR
                            Pre-processed data directory

    optional arguments:
    -h, --help            show this help message and exit
    --warm-start-model {CHAP_ACT_AUSDIAB}
                            Pre-trained model to warm start the training
    --learning-rate LEARNING_RATE
                            Learning rate for training the model
    --num-epochs NUM_EPOCHS
                            Number of epochs to train the model
    --batch-size BATCH_SIZE
                            Training batch size
    --amp-factor AMP_FACTOR
                            Factor to increase the number of neurons in the CNN
                            layers
    --cnn-window-size CNN_WINDOW_SIZE
                            CNN window size in seconds on which the predictions to
                            be made
    --bi-lstm-window-size BI_LSTM_WINDOW_SIZE
                            BiLSTM window size in minutes on which the predictions
                            to be smoothed
    --shuffle-buffer-size SHUFFLE_BUFFER_SIZE
                            Training data shuffle buffer size in terms of number
                            of records
    --training-data-fraction TRAINING_DATA_FRACTION
                            Percentage of subjects to be used for training
    --validation-data-fraction VALIDATION_DATA_FRACTION
                            Percentage of subjects to be used for validation
    --testing-data-fraction TESTING_DATA_FRACTION
                            Percentage of subjects to be used for testing
    --model-checkpoint-path MODEL_CHECKPOINT_PATH
                            Path where the trained model will be saved
    --num-classes NUM_CLASSES
                            Number of classes in the training dataset
    --class-weights CLASS_WEIGHTS
                            Class weights for loss aggregation
    --down-sample-frequency DOWN_SAMPLE_FREQUENCY
                            Downsample frequency in Hz for GT3X data
    --silent              Whether to hide info messages

**Model Selection:** Notice that this script relies on several hyperparameters required for training the model such as learning rate, batch size, and BiLSTM window size etc. The script comes with set of default values for these parameters. However, you may need to tweak these parameters for your dataset to get the best performance.

**Transfer Learning:** Instead of start training a model from scratch, you can also start with an existing model and tune it for your dataset. This is also called transfer learning and can be used to train a high-quality model with limited amount of training data. Currently we support transfer learning using the `CHAP_ACT_AUSDIAB` model. In order to use transfer learning pass the `--transfer-learning-model CHAP_ACT_AUSDIAB` option when invoking the `train_model.py` script. Note that with transfer learning only the training hyperparameters (e.g., batch size, learning rate, num epochs etc.) can be tweaked. The architectural hyperparameters (e.g., BiLSTM window size, CNN window size) will be set to the values of the source model. When transfer learning, it is recommeded to use a low learning rate (e.g., 0.00001) to avoid overfitting.

**Generating Prediction using a Custom Model:** After training your own model you can use it to generate predictions by passing the model checkpoint path to the `make_predictions.py` script as follows:

    python make_predictions.py --pre-processed-dir <pre-processed-dir> --predictions-dir <predictions-dir> --model-checkpoint-path <checkpoint-dir>    