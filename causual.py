from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch.fx

prompt = "Leaning too far forward causes the buttocks to stick out because"
hug_model = 'KnutJaegersberg/CausalLM-Platypus-14B'
tokenizer = AutoTokenizer.from_pretrained(hug_model)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
model = AutoModelForCausalLM.from_pretrained(hug_model)
outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

from datasets import load_dataset
data = load_dataset("open-llm-leaderboard/details_KnutJaegersberg__CausalLM-Platypus-14B",
    "harness_winogrande_5",
    split="latest",)
print(data[2])