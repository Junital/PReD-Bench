#!/bin/bash

LLM_PATH = "PATH_TO_LLM"

python XAG/finetune.py --llm_dir "$LLM_PATH"
python XAG/evaluate.py --llm_dir "$LLM_PATH"