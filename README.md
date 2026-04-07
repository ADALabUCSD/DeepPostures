CHAP 2.0 — Finetuning for Wrist & Hip Accelerometer Posture Classification
============================================================================

This repository contains the **PyTorch** codebase for classifying sedentary and activity postures from accelerometer data. It includes all models from the paper:

- **CHAP-ZS**: Zero-shot CNN-BiLSTM model
- **CHAP-FT**: Finetuned CNN-BiLSTM model for posture classification
- **CHAP-ViT**: Vision Transformer-based model variants (ViT-base, ViT-small, ViT-tiny)

All models support finetuning on both hip and wrist (iWatch) accelerometer datasets, including SOL/PASOS.

For the legacy TensorFlow code and earlier publications, see the `master` branch.


Repository Structure
--------------------

```
CHAP2/                      # Main finetuning pipeline
├── example_models.py       # Example: model instantiation, weight loading, inference
├── main_finetune.py        # Finetuning entry point
├── chap_model.py           # CNN-BiLSTM model definitions (CHAP-FT / CHAP-ZS)
├── models_vit.py           # Vision Transformer model definitions (CHAP-ViT)
├── vision_transformer.py   # ViT with Rotary Position Embedding (RoPE)
├── engine_finetune.py      # Training and evaluation loops
├── requirements.txt        # Python dependencies
├── util/                   # Utilities (learning rate, data, loss, etc.)
├── script/                 # Training launch scripts
│   ├── iwatch_vit.sh               # iWatch CHAP-ViT finetuning
│   ├── chap_ft_iwatch.sh           # iWatch CHAP-FT finetuning
│   ├── chap_ft_sol.sh              # SOL dataset CHAP-FT finetuning
│   └── chap_scratch_sol.sh         # SOL training from scratch
└── SUBMIT_RESULT/          # Submitted checkpoints and predictions
    ├── H/                          # Hip models
    │   ├── CHAP-FT/checkpoint/     # CHAP-FT weights
    │   └── CHAP-ZS/checkpoint/     # CHAP-ZS weights
    └── W/                          # Wrist models
        ├── CHAP-FT/checkpoint/
        └── CHAP-ZS/checkpoint/

MSSE_2021_pt/               # PyTorch port of CNN/BiLSTM baselines (CHAP 1.0)
support_files/              # Validation data and participant-level agreement files
```


Pre-Requisites
--------------
- Python 3.8+
- A GPU machine is strongly recommended for training.

Install dependencies:

```bash
pip install -r CHAP2/requirements.txt
```


Usage
-----

See `CHAP2/example_models.py` for a self-contained example of how to instantiate each model, load pretrained weights, and run inference. It covers CHAP 2.0 (CNN-BiLSTM), CHAP-ZS (CNN-BiLSTM + Attention), and CHAP-ViT.

Training scripts are located in `CHAP2/script/`. Example:

```bash
cd CHAP2
bash script/iwatch_vit.sh
```

See individual scripts for dataset paths and hyperparameter configurations.


Related Publications
--------------------
- **CHAP (MSSE 2021):** *The CNN Hip Accelerometer Posture (CHAP) Method for Classifying Sitting Patterns from Hip Accelerometers: A Validation Study in Older Adults*
- **JMPB 2021:** *Application of Convolutional Neural Network Algorithms for Advancing Sedentary and Activity Bout Classification* — [DOI](https://doi.org/10.1123/jmpb.2020-0016) | [Paper](https://adalabucsd.github.io/papers/2021_JMPB_CNN.pdf)


Acknowledgement
---------------
This work was supported by grant number R01DK114945 from the National Institute of Diabetes and Digestive and Kidney Diseases. It was also supported in part by a Hellman Fellowship, an NSF CAREER Award under award number 1942724, and a gift from VMware. The content is solely the responsibility of the authors and does not necessarily represent the views of any of these organizations. We thank the members of UC San Diego's Database Lab and Center for Networked Systems for their feedback on this work.