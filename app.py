import gradio as gr
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
import torch

# 加载模型
MODEL_PATH = "output/checkpoint-1500"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16  # 半精度加速
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


def generate_answer(question, temperature=0.7, max_length=200):
    try:
        prompt = f"问题：{question}\n答案："
        output = pipe(
            prompt,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True
        )[0]['generated_text']
        return output.split("答案：")[1].strip()
    except Exception as e:
        return f"生成错误：{str(e)}"


# 构建界面
with gr.Blocks() as demo:
    gr.Markdown("# 智能问答小助手")
    with gr.Row():
        with gr.Column():
            input_question = gr.Textbox(label="输入问题", placeholder="请输入您的问题...")
            temperature = gr.Slider(0.1, 1.0, value=0.7, label="温度参数")
            max_length = gr.Slider(50, 500, value=200, label="最大生成长度")
            submit_btn = gr.Button("提交")
        with gr.Column():
            output_answer = gr.Textbox(label="模型回答", interactive=False)

    submit_btn.click(
        fn=generate_answer,
        inputs=[input_question, temperature, max_length],
        outputs=output_answer
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)