import json
from datasets import Dataset
from transformers import AutoTokenizer

def load_data(file_path):
    """加载JSONL格式数据"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def preprocess_data(data, tokenizer, max_length=512):
    """将问答对转换为模型输入格式"""
    processed = []
    for item in data:
        prompt = f"问题：{item['instruction']}\n答案："
        inputs = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
        inputs["labels"] = tokenizer(
            item["output"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        ).input_ids
        processed.append(inputs)
    return Dataset.from_list(processed)

# 示例数据生成（train.jsonl）
"""
{"instruction": "什么是反向传播？", "output": "反向传播是一种..."}
{"instruction": "解释梯度下降", "output": "梯度下降是..."}
"""