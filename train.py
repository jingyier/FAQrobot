import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from utils.data_processing import load_data, preprocess_data

# 配置参数
MODEL_NAME = "Qwen/Qwen-7B"  # 或 "meta-llama/Llama-2-7b"
DATA_PATH = "data/train.jsonl"
OUTPUT_DIR = "output"
LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],  # Qwen的注意力层名称
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

# 加载模型与分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=dict(load_in_4bit=True)  # 4-bit量化
)
model = get_peft_model(model, LORA_CONFIG)

# 数据预处理
train_data = load_data(DATA_PATH)
dataset = preprocess_data(train_data, tokenizer)

# 训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,    # 根据显存调整
    gradient_accumulation_steps=4,     # 模拟更大批次
    learning_rate=2e-5,
    warmup_steps=100,
    num_train_epochs=3,
    logging_dir="./logs",
    save_strategy="steps",
    save_steps=500,
    fp16=True,                        # 启用混合精度
    gradient_checkpointing=True        # 减少显存
)

# 开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()