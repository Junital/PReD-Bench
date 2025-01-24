import os
import re
import argparse
import pandas as pd
from tqdm import tqdm
from predbench.evaluate import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel, PeftConfig

parser = argparse.ArgumentParser(description="Evaluation for fine-tuned framework.")
parser.add_argument('--llm_dir', type=str, required=True, help="The path for pretrained large language models (like llama3).")
parser.add_argument('--task_name', type=str, default="xag", help="The task name for this fine-tuning.")
args = parser.parse_args()

dataset = load_dataset("json", data_files=f"./XAG/data/{args.task_name}_test_finetune.json")
dataset = dataset["train"]

def format_input(batch):
    return {"input": [f"<s>[INST] {item} [/INST] " for item in batch["input"]]}

dataset = dataset.map(format_input, batched=True, remove_columns=["input"])


# load base LLM model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(args.llm_dir,  load_in_8bit=True,  device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(args.llm_dir)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

total_data = {
    "model": [],
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": []
}

def get_checkpoints(folder_path):
    # 定义正则表达式，用于匹配 "checkpoint-数字"
    pattern = re.compile(r'^checkpoint-(\d+)$')
    checkpoints = []
    
    # 遍历文件夹中的内容
    for name in os.listdir(folder_path):
        # 检查是否是文件夹并匹配正则
        if os.path.isdir(os.path.join(folder_path, name)):
            match = pattern.match(name)
            if match:
                # 提取数字并添加到结果列表
                checkpoints.append(int(match.group(1)))
    
    return checkpoints

if not os.path.exists("./output"):
    os.makedirs("./output")

for number in tqdm(get_checkpoints(f"./{args.task_name}")):
    peft_model_id = f"./{args.task_name}/checkpoint-{number}/"
    config = PeftConfig.from_pretrained(peft_model_id)

    # Load the Lora model
    model = PeftModel.from_pretrained(base_model, peft_model_id, device_map={"":0})

    pred = []
    gt = dataset["output"]

    for x in dataset["input"]:
        inputs = tokenizer(x, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].cuda()
        attention_mask = inputs["attention_mask"].cuda()
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=False, max_new_tokens=1)
        generated_tokens = outputs.detach().cpu().numpy()[:, input_ids.shape[-1]:]
        ans = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        pred.append(classify_text(ans))

    gt_num = [1 if x == "Yes" else 0 for x in gt]
    pred_num = [1 if x == "Yes" else 0 for x in pred]

    data = pd.DataFrame({"Ground Truth": gt_num, "Prediction": pred_num})
    data.to_json(f"./output/{args.task_name}_finetune_{number}_steps.json", orient="index")

    accu, prec, rec, f1 = calculate_metrics(gt_num, pred_num)
    total_data["model"].append(f"{args.task_name} {number}")
    total_data["accuracy"].append(accu)
    total_data["precision"].append(prec)
    total_data["recall"].append(rec)
    total_data["f1"].append(f1)

total_df = pd.DataFrame(total_data)
print(total_df)
total_df.to_csv(f"./output/{args.task_name}.csv", index=False)