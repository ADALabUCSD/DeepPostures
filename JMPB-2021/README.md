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

- **(Optional) Events Data**: Optionally, you can also provide ActivPal events data, especially if you wish to train your own models, for each subjects as a single .csv file. These files should also be named in the **<subject_id>.csv** format and files for all subjects should be put in the same directory. First few lines of a sample csv file are as follows:
    ~~~
    StartTime,EndTime,Behavior
    2014-05-07 09:47:23,2014-05-07 09:48:21,sitting
    2014-05-07 09:48:22,2014-05-07 09:48:26,standing
    2014-05-07 09:48:27,2014-05-07 09:49:03,stepping
    2014-05-07 09:49:04,2014-05-07 09:49:04,standing
    2014-05-07 09:49:05,2014-05-07 09:49:11,sitting
    ~~~

More details on how to process ActivPal events data files can be found in our paper.

## Pre-Processing Data
First, you need to create pre-processed data from the source data. To do this invoke the `pre_process_data.py` script as follows:

    python pre_process_data.py --gt3x-dir <gt3x_data_dir> --activpal-dir <activpal_data_dir> --pre-processed-dir <output_dir>

Complete usage details of this script are as follows:

    usage: pre_process_data.py [-h] --gt3x-dir GT3X_DIR --pre-processed-dir
                           PRE_PROCESSED_DIR [--activpal-dir ACTIVPAL_DIR]
                           [--window-size WINDOW_SIZE]
                           [--gt3x-frequency GT3X_FREQUENCY]
                           [--activpal-label-map ACTIVPAL_LABEL_MAP]
                           [--silent]

    Argument parser for preprocessing the input data.

    required arguments:
    --gt3x-dir GT3X_DIR   GT3X data directory
    --pre-processed-dir PRE_PROCESSED_DIR
                            Pre-processed data directory

    optional arguments:
    -h, --help            show this help message and exit
    --activpal-dir ACTIVPAL_DIR
                            ActivPAL data directory
    --window-size WINDOW_SIZE
                            Window size in seconds on which the predictions to be
                            made
    --gt3x-frequency GT3X_FREQUENCY
                            GT3X device frequency in Hz
    --activpal-label-map ACTIVPAL_LABEL_MAP
                            ActivPal label vocabulary
    --silent              Whether to hide info messages

## Generating Predictions
You can use the released pre-trained models to generate predictions using your own data. To do so invoke the `make_predictions.py` as follows:

    python make_predictions.py --pre-processed-dir <pre-processed-dir> --predictions-dir <predictions-dir>

Complete usage details of this script are as follows:

    usage: make_predictions.py [-h] --pre-processed-dir PRE_PROCESSED_DIR
                            [--predictions-dir PREDICTIONS_DIR]
                            [--batch-size BATCH_SIZE]
                            [--num-classes NUM_CLASSES]
                            [--window-size WINDOW_SIZE]
                            [--gt3x-frequency GT3X_FREQUENCY] [--no-label]
                            [--model-checkpoint-path MODEL_CHECKPOINT_PATH]
                            [--remove-gravity] [--silent]

    Argument parser for generating model predictions.

    required arguments:
    --pre-processed-dir PRE_PROCESSED_DIR
                            Pre-processed data directory

    optional arguments:
    -h, --help            show this help message and exit
    --predictions-dir PREDICTIONS_DIR
                            Training batch size
    --batch-size BATCH_SIZE
                            Training batch size
    --num-classes NUM_CLASSES
                            Number of classes in the training dataset
    --window-size WINDOW_SIZE
                            Window size in seconds on which the predictions to be
                            made
    --gt3x-frequency GT3X_FREQUENCY
                            GT3X device frequency in Hz
    --no-label            Whether to not output the label
    --model-checkpoint-path MODEL_CHECKPOINT_PATH
                            Path where the trained model will be saved
    --remove-gravity      Whether to remove gravity from accelerometer data
    --silent              Whether to hide info messages

## Training Your Own Model
To train your own model invoke the `train_model.py` as follows:

    python --pre-processed-dir <pre-processed-dir> --model-checkpoint-path <checkpoint-dir>

Complete usage details of this script are as follows:

    usage: train_model.py [-h] --pre-processed-dir PRE_PROCESSED_DIR
                        [--learning-rate LEARNING_RATE]
                        [--num-epochs NUM_EPOCHS] [--batch-size BATCH_SIZE]
                        [--dropout-rate DROPOUT_RATE]
                        [--shuffle-buffer-size SHUFFLE_BUFFER_SIZE]
                        [--training-data-fraction TRAINING_DATA_FRACTION]
                        [--validation-data-fraction VALIDATION_DATA_FRACTION]
                        [--testing-data-fraction TESTING_DATA_FRACTION]
                        [--model-checkpoint-path MODEL_CHECKPOINT_PATH]
                        [--window-size WINDOW_SIZE]
                        [--gt3x-frequency GT3X_FREQUENCY]
                        [--num-classes NUM_CLASSES]
                        [--class-weights CLASS_WEIGHTS] [--remove-gravity]
                        [--silent]

    Argument parser for training CNN model.

    required arguments:
    --pre-processed-dir PRE_PROCESSED_DIR
                            Pre-processed data directory

    optional arguments:
    -h, --help            show this help message and exit
    --learning-rate LEARNING_RATE
                            Learning rate for training the model
    --num-epochs NUM_EPOCHS
                            Number of epochs to train the model
    --batch-size BATCH_SIZE
                            Training batch size
    --dropout-rate DROPOUT_RATE
                            Dropout rate during training
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
    --window-size WINDOW_SIZE
                            Window size in seconds on which the predictions to be
                            made
    --gt3x-frequency GT3X_FREQUENCY
                            GT3X device frequency in Hz
    --num-classes NUM_CLASSES
                            Number of classes in the training dataset
    --class-weights CLASS_WEIGHTS
                            Class weights for loss aggregation
    --remove-gravity      Whether to remove gravity from accelerometer data
    --silent              Whether to hide info messages

Notice that this script relies on several hyperparameters required for training the model such as learning rate, batch size, and number of training epochs etc. The script comes with set of default values for these parameters. However, you may need to tweak these parameters for your dataset to get the best performance.

After training your own model you can use it to generate predictions by passing the model checkpoint path to the `make_predictions.py` script as follows:

    python make_predictions.py --pre-processed-dir <pre-processed-dir> --predictions-dir <predictions-dir> --model-checkpoint-path <checkpoint-dir>
