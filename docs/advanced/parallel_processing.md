---
layout: default
title: 2. Parallel Processing
parent: Advanced Usages
nav_order: 3
---

# Parallelizing the Code Execution

The `pre_process_data.py` script is operated in sequential mode (i.e., it will finish pre-processing data for a single subject before proceeding to the next subject). If the user has access to a multi-core machine, we recommend that they utilize the `--mp` toggle of the script to enable multiprocessing. To do so, users will simply add `--mp <number of workers>` to the script invocation command. It is recommended to set the number to the number of CPU cores available in a machine. However, the user has to ensure that the aggregate memory consumption does not exceed the total available system memory. Example:

```
python pre_process_data.py --gt3x-dir ./gt3x_dir --pre-processed-dir ./pre-processed --sleep-logs-file ./sleep_logs.csv --mp 4
```