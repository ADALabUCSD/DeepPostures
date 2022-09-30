---
layout: default
title: 3. Data Compression
parent: Advanced Usages
nav_order: 4
---

# Data Compression

The GT3X RAW files are large in size, and transmitting them through disk or network could take a lot of time and substantial storage space. To mitigate these issues, we recommend compressing these files using gzip as the files are highly compressible (we reduced 800GB of dataset to 40GB). Then use the `--gzipped` toggle of `pre_process_data.py` to consume the gzipped documents directly. Note: only the RAW files can be passed as gzipped, each RAW file must be compressed individually, and the extension must be `.csv.gz`.