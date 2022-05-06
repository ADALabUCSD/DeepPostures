---
layout: default
title: 1. Dependencies
parent: Getting Started
nav_order: 1
---

# Dependencies

We recommend first [installing Anaconda](https://www.anaconda.com/) and then running the following commands to setup the environment. We also recommend using a machine that has GPU support, specially if you plan to train your own models. A CPU machine can be used if the goal is using pre-trained models to predict posture.

Now [open a terminal](https://docs.anaconda.com/anaconda/user-guide/getting-started/#write-a-python-program-using-anaconda-prompt-or-terminal) and run commands:

```bash
    conda env create -f INFRA/CONDA/deep_postures_gpu_env.yml # for cpu use INFRA/CONDA/deep_postures_cpu_env.yml
    conda activate deep_postures
```

If the above doesn't work, you can do it manually.

```bash
    conda create -n deep_postures python=3.6
    conda activate deep_postures
    conda install "tensorflow-gpu>=1.13.0,<2.0" # for cpu use "tensorflow>=1.13.0,<2.0"
    conda install pandas numpy scipy h5py
```

**Now, move into the MSSE-2021 folder where our newest model is hosted:**


```bash
    cd MSSE-2021
```


