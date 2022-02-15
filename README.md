DeepPostures
============

This repository contains the code artifacts released as part of the following publications:

- /JMPB-2021 : **Application of Convolutional Neural Network Algorithms for Advancing Sedentary and Activity Bout Classification**, Journal for the Measurement of Physical Behaviour, [DOI](https://doi.org/10.1123/jmpb.2020-0016)|[Paper](https://adalabucsd.github.io/papers/2021_JMPB_CNN.pdf)
- /MSSE-2021 : **The CNN Hip Accelerometer Posture (CHAP) Method for Classifying Sitting Patterns from Hip Accelerometers: A Validation Study in Older Adults**


Pre-Requisites
--------------
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


Instructions
------------
- Clone the repository using git `git clone https://github.com/ADALabUCSD/DeepPostures.git`
- Navigate to the code directory of the publication you want to explore and follow the instructions in the README file there.
- If you face any problems/issues, please create an issue in GitHub issue tracker.


Acknowledgement
---------------
This work was supported by grant number R01DK114945 from the National Institute of Diabetes and Digestive and Kidney Diseases. It was also supported in part by a Hellman Fellowship, an NSF CAREER Award under award number 1942724, and a gift from VMware. The content is solely the responsibility of the authors and does not necessarily represent the views of any of these organizations. We thank the members of UC San Diego's Database Lab and Center for Networked Systems for their feedback on this work.
