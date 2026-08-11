---
layout: default
title: 0. Dependencies
parent: Getting Started
grand_parent: CHAP2.0
nav_order: 0
---

# Dependencies

We recommend first [installing Anaconda](https://www.anaconda.com/) and then
creating a Python environment for CHAP2.0. We also recommend using a machine
that has GPU support, especially if you plan to fine-tune your own model. A CPU
machine can be used if the goal is generating predictions with a pre-trained
checkpoint.

Now [open a terminal](https://docs.anaconda.com/anaconda/user-guide/getting-started/#write-a-python-program-using-anaconda-prompt-or-terminal)
and run commands:

```bash
    conda create -n chap2 python=3.10 -y
    conda activate chap2
```

Then install the CHAP2.0 requirements from the repository root:

```bash
    cd CHAP2
    pip install -r requirements.txt
```

Run the CHAP2.0 scripts from inside the `CHAP2/` folder.
