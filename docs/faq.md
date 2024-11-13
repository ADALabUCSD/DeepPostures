---
layout: default
title: FAQs
nav_order: 9
---

# FAQs
{: .no_toc }


<!-- - **Q: My TensorFlow does not expect the model format.**

    **A:** To use the library you will need specific versions of TensorFlow, please refer to [Getting Started]({{ site.baseurl }}{% link getting_started/dependencies.md %}) -->

- **Q: What Python version should I have?**

    **A:** The bare minimum you should have is Python 3. We tested and recommend using Python 3.11 as other Python versions may cause problems with certain dependencies.

- **Q: The script runs very slow on my machine. What should I do?**

    **A:** First you need to isolate the bottleneck and the cause of slowness. If it's from data preprocessing, consider parallel processing and follow [guide]({{ site.baseurl }}{% link advanced/parallel_processing.md %}). If it's the predicting/learning part, then you should try to enable GPU (if you have one) support by installing the GPU-version of tensorflow, or consider using a more powerful machine or a machine with GPU.

- **Q: The data is huge; storing and moving it takes a lot space and time.**

    **A:** We recommend enable data compression following [guide]({{ site.baseurl }}{% link advanced/compression.md %}).

- **Q: My job was running on Kubernetes cluster and got killed with status OOMKilled during CHAP pre_process_data.py stage.**

    **A:** One potential reasons could be extremely large ActiGraph 30Hz Raw file, such as > 4GB. Try to remove the large file and re-run.