---
layout: default
title: 3. Generating Predictions
parent: Getting Started
---

# Generating Predictions from Pre-Trained Models
{: .no_toc }

You can use the released [pre-trained models]({{ site.baseurl }}{% link dataset_and_models.md %}#Pre-trained Models) to generate predictions using your own data. To do so invoke the `make_predictions.py` as follows:
    
    python make_predictions.py --pre-processed-dir <pre_processed_data_dir>

Complete usage details of this script are as follows:

    usage: make_predictions.py [-h] --pre-processed-dir PRE_PROCESSED_DIR
                            [--model {CHAP_A,CHAP_B,CHAP_C,CHAP,CHAP_ALL_ADULTS,CHAP_CHILDREN}]
                            [--predictions-dir PREDICTIONS_DIR]
                            [--no-segment] [--output-label]
                            [--silent]

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
