from transformers import pipeline
from utils.data_processing import load_data
import numpy as np
from rouge import Rouge

# 加载测试数据
test_data = load_data("data/test.jsonl")

# 加载微调后的模型
model_path = "output/checkpoint-1500"  # 替换为实际路径
pipe = pipeline("text-generation", model=model_path, device=0)

# 计算ROUGE-L
rouge = Rouge()
predictions, references = [], []

for item in test_data[:10]:  # 抽样评估
    prompt = f"问题：{item['instruction']}\n答案："
    generated = pipe(
        prompt,
        max_new_tokens=200,
        temperature=0.7
    )[0]['generated_text']
    answer = generated.split("答案：")[1].strip()

    predictions.append(answer)
    references.append(item["output"])

# 计算指标
rouge_scores = rouge.get_scores(predictions, references, avg=True)
print(f"ROUGE-L: {rouge_scores['rouge-l']['f']:.3f}")