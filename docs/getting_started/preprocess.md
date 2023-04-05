---
layout: default
title: 2. Data Preprocessing
parent: Getting Started
---

# Data Preprocessing
{: .no_toc }

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

**Note:** If you use our pre-trained models for generating predictions, please keep the `--window-size` config unmodified, as this is what our models were trained on and cannot be changed. You can modify this if you train your own model (see [instructions]({{ site.baseurl }}{% link advanced/training.md %}), the corresponding config is `--cnn-window-size` ).

### Data format
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

    **Note:** The expected .csv file must have the headers (first few lines) in the above format. Any other data format needs to be converted first. The data is usually very compressible, so we recommend compressing with gzip or equivalent before network transmission (don't forget to uncompress).
    
    **Note:** Our pre-trained models work and make predictions on time-series windows. If your data length is not an exact multiple of the window size, the last a few minutes that is not enough to make up a whole window will be dropped. This simply truncates the participant’s data period by a couple of minutes on average, resulting in slightly reduced wear time by a couple of minutes. This is usually not an issue in most scenarios, but if it matters for your task, you can choose a imputation scheme during inference as described [here]({{ site.baseurl }}{% link getting_started/inference.md %}). Alternatively, you can implement other imputation strategies as appropriate for your study directly on the .csv data. Note the predictions on this last window would be somewhat unreliable due to the missing data.

    The CHAP development team recommends running CHAP on each participant’s full data collection period prior to removing periods of non-wear, sleep, etc. Using this approach, there are no gaps in a participant’s data from the beginning to end of their data collection period, so CHAP will run predictions on the entire data collection period. The CHAP time-series prediction window will only impact the final few minutes of the data period (at most), as long as non-wear, sleep, etc. are removed after running CHAP rather than before.

    

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

### Speed up
If the data is too large and takes long time for network transmission, we recommend enable data compression following [compression guide]({{ site.baseurl }}{% link advanced/compression.md %}). If the preprocessing is taking too much time, consider parallel processing and follow [parallel guide]({{ site.baseurl }}{% link advanced/parallel_processing.md %}).
