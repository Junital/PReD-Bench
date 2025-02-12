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

Download the LLaMA 3 pretrained models to your preferred directory. Update the `LLM_PATH` in the `STRIDE.sh` script to point to the directory where the models are stored.

Execute the following command to run the `STRIDE` script:

```bash
bash STRIDE.sh
```

## Dataset Card
Huggingface link: https://huggingface.co/datasets/Junital/PReD-Bench

task_categories:
- token-classification
- question-answering

language:
- en

pretty_name: PReD-Bench

size_categories:
- 1K<n<10K

You can use the dataset: 
``` python
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("Junital/PReD-Bench")
```