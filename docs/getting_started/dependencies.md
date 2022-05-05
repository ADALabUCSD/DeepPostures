---
layout: default
title: 1. Dependencies
parent: Getting Started
nav_order: 1
---

# Dependencies

We recommend first [installing Anaconda](https://docs.anaconda.com/anaconda/install/) and then running the following commands to setup the environment. We also recommend using a machine that has GPU support, specially if you plan to train your own models. A CPU machine can be used if the goal is using pre-trained models to predict posture.

    conda env create -f INFRA/CONDA/deep_postures_gpu_env.yml # for cpu use INFRA/CONDA/deep_postures_cpu_env.yml
    conda activate deep_postures


Alternatively, you can use conda to install Python 3 and use `pip` to install the following rerquired packages.
    
    conda create -n deep_postures python=3.6
    conda activate deep_postures
    python -m pip install "tensorflow-gpu>=1.13.0,<2.0" # for cpu use "tensorflow>=1.13.0,<2.0"
    python -m pip install pandas
    python -m pip install numpy
    python -m pip install scipy
    python -m pip install h5py
