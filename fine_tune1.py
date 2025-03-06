from datasets import load_dataset
from unsloth import FastLlamaModel  # Ajuste conforme seu modelo
from transformers import TrainingArguments
from trl import SFTTrainer
import torch
max_seq_length_param = 512
# Carrega o dataset específico
ds = load_dataset(
    "json",
    data_files="https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl",
    split="train[:1%]"
)

# from transformers import BitsAndBytesConfig

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

# model, tokenizer = FastLlamaModel.from_pretrained(
#     model_name="unsloth/llama-3-8b",
#     max_seq_length=max_seq_length_param,
#     quantization_config=quantization_config,
#     device_map="auto",
#     max_memory={0: "4GB"},  # Limita a GPU a 4GB (ajuste conforme necessário)
# )

model, tokenizer = FastLlamaModel.from_pretrained(
    model_name="unsloth/llama-3-8b",
    max_seq_length=max_seq_length_param,
    dtype=torch.float32,  # FP32 no CPU
    device_map="cpu",     # ✅ Força uso da CPU
)

# Configuração PEFT (exemplo)
from peft import LoraConfig
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
)

# Prepara o modelo para treinamento com PEFT
model = model.prepare_for_training(
    peft_config=peft_config,
    train_dataset=ds,
    max_seq_length=max_seq_length_param,
)

# Configuração do Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    max_seq_length=max_seq_length_param,
    tokenizer=tokenizer,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=60,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        output_dir='./logs',
        optim='adamw_8bit',
        weight_decay=0.01,
        lr_scheduler_type='linear',
        seed=3047
    ),
)

trainer.train()