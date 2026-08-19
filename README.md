DeepPostures
============
**New** Check out our website at [url](https://adalabucsd.github.io/DeepPostures/) for tutorials, demos, and many more! 

This repository contains code artifacts for CHAP posture classification work:

- `MSSE-2021/`: **CHAP1.0 PyTorch implementation for hip accelerometer sitting pattern classification**, released with *The CNN Hip Accelerometer Posture (CHAP) Method for Classifying Sitting Patterns from Hip Accelerometers: A Validation Study in Older Adults*.
- `CHAP2/`: **CHAP2.0 PyTorch code for CHAP zero-shot prediction and fine-tuning on wrist/hip accelerometer data**, including iWatch and SOL/PASOS workflows.

**We strongly suggest starting with our website and then using the folder that matches your workflow.** Use `MSSE-2021/` for the original CHAP pipeline and `CHAP2/` for CHAP zero-shot/fine-tuning workflows.

> The current codebase has been migrated to PyTorch. To access the previous TensorFlow implementation switch to `tensorflow` branch. The tensorflow branch contains implementation for both `/JMPB-2021` and `/MSSE-2021` 

Pre-Requisites
--------------
We recommend first [installing Anaconda](https://docs.anaconda.com/anaconda/install/) and then running the following commands to setup the environment. We also recommend using a machine that has GPU support, specially if you plan to train your own models. A CPU machine can be used if the goal is using pre-trained models to predict posture.

    conda env create -f INFRA/CONDA/deep_postures_pt_gpu.yml # for cpu use INFRA/CONDA/deep_postures_pt_cpu.yml
    conda activate deep_postures


Alternatively, you can use conda to install Python 3 and use `pip` to install the following rerquired packages.
    
    conda create -n deep_postures_pytorch python=3.11
    conda activate deep_postures_pytorch
    pip install torch==2.4.1
    pip install numpy
    pip install pandas
    pip install scipy
    pip install h5py
    pip install scikit-learn==1.5.2
    pip install tqdm==4.66.6

For CHAP2.0 workflows, install the base environment above and then install the additional packages listed in `CHAP2/requirements.txt`.


Instructions
------------
- Clone the repository using git `git clone https://github.com/ADALabUCSD/DeepPostures.git`
- Navigate to the code directory of the workflow you want to explore and follow the instructions in the README file there. The `main` branch contains the PyTorch implementation for `MSSE-2021` and CHAP2.0 workflows. The `tensorflow` branch contains the previous TensorFlow implementation for both `MSSE-2021` and `JMPB-2021`.
- If you face any problems/issues, please create an issue in GitHub issue tracker.


Acknowledgement
---------------
This work was supported by grant number R01DK114945 from the National Institute of Diabetes and Digestive and Kidney Diseases, and by grant number 1R01HL168535 from the National Heart, Lung and Blood Institute. It was also supported in part by a Hellman Fellowship, an NSF CAREER Award under award number 1942724, and a gift from VMware. The content is solely the responsibility of the authors and does not necessarily represent the views of any of these organizations. We thank the members of UC San Diego’s Database Lab and Center for Networked Systems for their feedback on this work.
