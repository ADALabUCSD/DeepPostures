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
    conda env create -f INFRA/CONDA/deep_postures_pt_gpu.yml # for cpu use INFRA/CONDA/deep_postures_pt_cpu.yml
    conda activate deep_postures
```

If the above doesn't work, you can do it manually.

```bash
    conda create -n deep_postures_pytorch python=3.11
    conda activate deep_postures_pytorch
    pip install torch==2.4.1
    pip install numpy
    pip install pandas
    pip install scipy
    pip install h5py
    pip install scikit-learn==1.5.2
    pip install tqdm==4.66.6
```

**Now, move into the MSSE-2021 folder where our newest model is hosted:**


```bash
    cd MSSE-2021
```


