# Table of Contents
- [Table of Contents](#table-of-contents)
  - [Data](#data)
  - [Pre-Processing Data](#pre-processing-data)
  - [Generating Predictions from Pre-Trained Models](#generating-predictions-from-pre-trained-models)
  - [Training Your Own Model](#training-your-own-model)
  - [Generating Predictions using a Custom-Trained Model](#generating-predictions-using-a-custom-trained-model)
  - [Parallelizing the Code Execution](#parallelizing-the-code-execution)
  - [Converting existing Tensorflow Weights to PyTorch](#parallelizing-the-code-execution)
  
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

    **Note:** The pre-processing function expects, by default, an Actigraph RAW file at 30hz with normal filter. If you have Actigraph RAW data at a different Hz level, please specify in the pre_process_data function using the gt3x-frequency parameter.

- **(Optional) Valid Days File**: A .csv file indicating which dates are valid (concurrent wear days) for all subjects. Each row is subject id, date pair. The header should be of the from `ID,Date.Valid.Day`.  Date values should be formatted in `%m/%d/%Y` format. A sample valid days file is shown below.

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

    python pre_process_data.py --gt3x-dir <gt3x_data_dir> --pre-processed-dir <output_dir>

Complete usage details of this script are as follows:

    usage: pre_process_data.py [-h] --gt3x-dir GT3X_DIR --pre-processed-dir PRE_PROCESSED_DIR [--valid-days-file VALID_DAYS_FILE]
                               [--sleep-logs-file SLEEP_LOGS_FILE] [--non-wear-times-file NON_WEAR_TIMES_FILE]
                               [--activpal-dir ACTIVPAL_DIR] [--n-start-id N_START_ID] [--n-end-id N_END_ID]
                               [--expression-after-id EXPRESSION_AFTER_ID] [--window-size WINDOW_SIZE]
                               [--gt3x-frequency GT3X_FREQUENCY] [--down-sample-frequency DOWN_SAMPLE_FREQUENCY]
                               [--activpal-label-map ACTIVPAL_LABEL_MAP] [--silent] [--mp MP] [--gzipped]

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
                            The index of the starting character of the ID in gt3x file names. Indexing starts with 1. If specified
                            `n_end_ID` should also be specified. I both `n_start_ID` and `expression_after_ID` is specified, the
                            latter will be ignored
      --n-end-id N_END_ID   The index of the ending character of the ID in gt3x file names
      --expression-after-id EXPRESSION_AFTER_ID
                            String or list of strings specifying different string spliting character that should be used to identify
                            the ID from gt3x file name. The first split will be used as the file name
      --window-size WINDOW_SIZE
                            Window size in seconds on which the predictions to be made (default: 10)
      --gt3x-frequency GT3X_FREQUENCY
                            GT3X device frequency in Hz (default: 30)
      --down-sample-frequency DOWN_SAMPLE_FREQUENCY
                            Downsample frequency in Hz for GT3X data (default: 10)
      --activpal-label-map ACTIVPAL_LABEL_MAP
                            ActivPal label vocabulary (default: {"0": 0, "1": 1, "2": 1})
      --silent              Whether to hide info messages
      --mp MP               Number of concurrent workers. Hint: set to the number of cores (default: None)
      --gzipped             Whether the raw data is gzipped or not. Hint: extension should be .csv.gz (default: False)

**Note:** If you use our pre-trained models for generating predictions, please keep the `--window-size` config unmodified, as this is what our models were trained on and cannot be changed. You can modify this if you train your own model (see instructions below, the corresponding config is `--cnn-window-size` ).

## Generating Predictions from Pre-Trained Models

You can use the released pre-trained models to generate predictions using your own data. To do so invoke the `make_predictions.py` as follows:
    
    python make_predictions.py --pre-processed-dir <pre_processed_data_dir>

Complete usage details of this script are as follows:

    usage: make_predictions.py [-h] --pre-processed-dir PRE_PROCESSED_DIR
                            [--model {CHAP_A,CHAP_B,CHAP_C,CHAP,CHAP_ALL_ADULTS,CHAP_CHILDREN}]
                            [--predictions-dir PREDICTIONS_DIR]
                            [--no-segment] [--output-label]
                            [--silent] [--padding {drop,zero,wrap}]
                            [--batch-size BATCH_SIZE] [--amp-factor AMP_FACTOR]
                            [--num-classes NUM_CLASSES]

    Argument parser for generating model predictions.

    required arguments:
    --pre-processed-dir PRE_PROCESSED_DIR
                            Pre-processed data directory

    optional arguments:
    -h, --help            show this help message and exit
    --model {CHAP_A,CHAP_B,CHAP_C,CHAP,CHAP_ALL_ADULTS,CHAP_CHILDREN}
                            Pre-trained prediction model name (default:
                            CHAP_ALL_ADULTS)
    --predictions-dir PREDICTIONS_DIR
                            Predictions output directory (default: ./predictions)
    --no-segment          Do not output segment number
    --output-label        Whether to output the actual label
    --silent              Whether to hide info messages
    --padding {drop,zero,wrap}
                        Padding scheme for the last part of data that does not
                        fill a whole lstm window (default: drop)
    --batch-size BATCH_SIZE
                            Inference batch size (default: 16)
    --amp-factor AMP_FACTOR
                            Factor to increase the number of neurons in the CNN
                            layers (default: 2)
    --num-classes NUM_CLASSES
                            Number of classes in the training dataset (default: 2)


We currently support several pre-trained models that can be used to generate predictions. They have been trained on different training datasets, which have different demographics. The recommended and default model is the `CHAP_ALL_ADULTS` model. However, users can change the pre-trained model to better match their needs using the `--model` option. Below we provide a summary of the available pre-trained models and the characteristics of the datasets that they were trained on.

| Model                                               | Training Dataset    |
|-----------------------------------------------------|---------------------|
|CHAP_ALL_ADULTS  (default and recommended)           | ACT + AUSDIAB       |
|CHAP_A                                               | ACT                 |
|CHAP_B                                               | ACT                 |
|CHAP_C                                               | ACT                 |
|CHAP (ensemble of A, B, and C)                       | ACT                 |
|CHAP_CHILDREN                                        | PHASE               |

|Training Dataset | Description                                             |
|-----------------|---------------------------------------------------------|
|ACT              | ACT is a cohort of community dwelling older adults age 65+. At time of accelerometer wear, the sample had a mean age of 76.7 years and was approximately 59% female and 90% non-Hispanic White.|
|AUSDIAB          | AUSDIAB is a cohort of older adults. At time of accelerometer wear, the sample had a mean age of 58.3 years and was approximately 56% female.| 
|PHASE            | PHASE is a cohort of children. At time of accelerometer wear, the sample had a mean age of x years and was approximately x female.| 

## Training Your Own Model
To train your own model, invoke the `train_model.py` as follows:

    python train_model.py --pre-processed-dir <pre-processed-dir> --model-checkpoint-path <checkpoint-dir>

Complete usage details of this script are as follows:

    usage: train_model.py [-h] --pre-processed-dir PRE_PROCESSED_DIR
                      [--transfer-learning-model {CHAP_ALL_ADULTS,CHAP_AUSDIAB,NONE}]
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
      --transfer-learning-model {CHAP_ALL_ADULTS,CHAP_AUSDIAB,NONE}
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

## Generating Predictions using a Custom-Trained Model
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
      --model {CHAP_A,CHAP_B,CHAP_C,CHAP,CHAP_ALL_ADULTS,CHAP_CHILDREN,CHAP_AUSDIAB}
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
                            Factor to increase the number of neurons in the CNN
                            layers (default: 2)
      --num-classes NUM_CLASSES
                            Number of classes in the training dataset (default: 2)


## Parallelizing the Code Execution
The `pre_process_data.py` script is operated in sequential mode (i.e., it will finish pre-processing data for a single subject before proceeding to the next subject). If the user has access to a multi-core machine, we recommend that they utilize the `--mp` toggle of the script to enable multiprocessing. To do so, users will simply add `--mp <number of workers>` to the script invocation command. It is recommended to set the number to the number of CPU cores available in a machine. However, the user has to ensure that the aggregate memory consumption does not exceed the total available system memory. Example:

```
python pre_process_data.py --gt3x-dir ./gt3x_dir --pre-processed-dir ./pre-processed --sleep-logs-file ./sleep_logs.csv --mp 4
```

## Data Compression
The GT3X RAW files are large in size, and transmitting them through disk or network could take a lot of time and substantial storage space. To mitigate these issues, we recommend compressing these files using gzip as the files are highly compressible (we reduced 800GB of dataset to 40GB). Then use the `--gzipped` toggle of `pre_process_data.py` to consume the gzipped documents directly. Note: only the RAW files can be passed as gzipped, each RAW file must be compressed individually, and the extension must be `.csv.gz`.


## Converting existing TensorFlow Weights to PyTorch
All the weights released as part of the TensorFlow implementation have been converted to a format which can be easily used by PyTorch.
If you have finetuned the tensorflow weights previouly and need to port them to PyTorch use the notebook `/tf2PtConversion/tf2PtWeightsConversion.ipynb` to convert them.
> Note this notebook is experimental and may require some additionally debugging on the end of the user.