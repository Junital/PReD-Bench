import argparse
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import DataCollatorForSeq2Seq, TrainingArguments
from trl import SFTTrainer

parser = argparse.ArgumentParser(description="Finetune for PReD-Bench.")
parser.add_argument('--llm_dir', type=str, required=True, help="The path for pretrained large language models (like llama3).")
parser.add_argument('--task_name', type=str, default="xag", help="The task name for this fine-tuning.")
parser.add_argument('--epoch', type=int, default=4, help="Epoch num for fine-tuning.")
args = parser.parse_args()

train_data = pd.read_json(f"./dataset/{args.task_name}_train_finetune.json")
text_json = []

for idx, row in train_data.iterrows():
    text_json.append({"text": "<s>[INST] " + row["input"] + " [/INST] " + row["output"]})
dataset = load_dataset('json', data_files={'train': text_json})

tokenizer = AutoTokenizer.from_pretrained(args.llm_dir)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(args.llm_dir, load_in_8bit=True, device_map="auto")

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

training_args = TrainingArguments(
    output_dir=args.task_name,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    max_grad_norm = 0.3,
    optim = "paged_adamw_32bit",
    warmup_ratio = 0.03,
    group_by_length = True,
    learning_rate=2e-4, # higher learning rate
    num_train_epochs=args.epoch,
    logging_dir=f"{args.task_name}/logs",
    logging_strategy="steps",
    logging_steps=200,
    save_strategy="steps",            # 每隔固定步数保存模型
    save_steps=200,                   # 每 100 步保存一次模型
    report_to="tensorboard",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    peft_config=lora_config,
    tokenizer=tokenizer,
    args=training_args
)

model.config.use_cache = False
trainer.train()
