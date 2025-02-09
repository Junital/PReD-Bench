#!/bin/bash

LLM_PATH='PATH_TO_LLM'

python -m XAG.finetune --llm_dir "$LLM_PATH"
python -m XAG.evaluate --llm_dir "$LLM_PATH"