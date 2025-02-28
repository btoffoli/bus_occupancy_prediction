import pandas as pd
from unsloth import FastLanguageModel

import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# mlname = 'unsloth/Qwen2.5-0.5B-bnb-4bit'
mlname = 'unsloth/tinyllama-bnb-4bit'
# mlname = 'microsoft/phi-2'

max_seq_length=512 #2048 #1024
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=mlname,
    max_seq_length=max_seq_length,
    dtype=torch.float16,
    load_in_4bit=True
)


print(model.device)



def generate_text(text: str):
    inputs = tokenizer(text, return_tensors="en").to('cuda:0')
    print(f"inputs: {inputs}")
    outputs = model.generate(**input, max_new_tokens=20)
    # print(f"outputs: {outputs}")
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"result: {result}")
    return result
    





if __name__ == '__main__':
    # Load the CSV file into a pandas DataFrame
    # df = pd.read_csv('data/vehicle_bus_stop_occupation_20250101_20250201-copy.csv')
    # print(df.columns)
    # print(df.dtypes)

    # print(df.head())
    # print(df[:10])

    generate_text("<human>Que horas são?\n<bot>:")

