
import os
from fine_tune1 import model, tokenizer
from peft import PeftModel, PeftConfig

pft_trained_model = PeftModel.from_pretrained(model, os.path.join(os.getcwd(), 'weight_models_sft'))


prompt = """Sabendo que está ensolarado e a temperatura é 29.00. Qual a lotação da linha 774I_P para o ponto 330726 para terça às 00:58?"""

inputs = tokenizer(prompt, return_tensors="pt").to('cuda:0')
print(f"inputs: {inputs}")
outputs = pft_trained_model.generate(**inputs, max_new_tokens=200)
# print(f"outputs: {outputs}")
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"result: {result}")