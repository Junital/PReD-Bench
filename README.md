# PReD-Bench

PReD-Bench: A benchmark for retraction risk detection using large language models.

## Installation

Set up a Python 3.10 environment using Anaconda:

```bash
conda create -n predbench python=3.10
conda activate predbench
```

Install the necessary packages for PReD-Bench using `pip`:

```bash
pip install -r requirements.txt
```

Download the LLaMA 3 pretrained models to your preferred directory. Update the `LLM_PATH` in the `xag.sh` script to point to the directory where the models are stored.

Execute the following command to run the `xag` script:

```bash
bash xag.sh
```
