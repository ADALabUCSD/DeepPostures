---
layout: default
title: 2. Parallel Processing
parent: Advanced Usages
nav_order: 3
---

# Parallelizing the Code Execution

The `pre_process_data.py` script is operated in sequential mode (i.e., it will finish pre-processing data for a single subject before proceeding to the next subject). If the user has access to a multi-core machine, we recommend that they run multiple invocations of the `pre_process_data.py` script where each invocation operates on a separate directory containing GT3X accelerometer data. To do so users will have to first split the accelerometer data into multiple sub-directories. The multiple invocations of `pre_process_data.py` script can still reuse the same other configurations files (e.g., sleep logs) and also the final pre-processed output directory. An example is shown below:


```
python pre_process_data.py --gt3x-dir ./gt3x_dir_1 --pre-processed-dir ./pre-processed --sleep-logs-file ./sleep_logs.csv &
python pre_process_data.py --gt3x-dir ./gt3x_dir_2 --pre-processed-dir ./pre-processed --sleep-logs-file ./sleep_logs.csv &
python pre_process_data.py --gt3x-dir ./gt3x_dir_3 --pre-processed-dir ./pre-processed --sleep-logs-file ./sleep_logs.csv &

wait
```

The number of parallel invocations to be performed for the `pre_process_data.py` script depends both on the number of cores available in the machine and the available memory. This number can be as high as the number of CPU cores available in a machine. However, the user has to ensure that the aggregate memory consumption by all invocations does not exceed the total available system memory.

The `make_predictions.py` script uses all available CPU cores in a machine. Thus, users **do not** need to explicitly parallelize this script's invocation.