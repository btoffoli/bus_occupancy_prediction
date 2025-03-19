from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
from dotenv import load_dotenv
import argparse
import logging

logger = logging.getLogger(__name__)



class BusOccupancyFineTune:

    """
        This is a fine tune class to manager the fine tune process for bus occupancy data.

    
    """
    

    def __init__(self, **kwargs):
        model_name = kwargs.get("model_name", "bert-base-uncased")
        self.datasets_path = kwargs.get("datasets_path", "./data")

        self.max_length = kwargs.get("max_length", 128)
        self.batch_size = kwargs.get("batch_size", 16)
        self.learning_rate = kwargs.get("learning_rate", 2e-5)
        self.num_epochs = kwargs.get("num_epochs", 3)
        self.output_dir = kwargs.get("output_dir", "./results")
        
        load_in_4bit = kwargs.get("load_in_4bit", False)
        bnb_4bit_compute_dtype = kwargs.get("bnb_4bit_compute_dtype", "float16")
        bnb_4bit_quant_type = kwargs.get("bnb_4bit_quant_type", "nf4")
        bnb_4bit_use_double_quant = kwargs.get("bnb_4bit_use_double_quant", True)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            # bnb_4bit_quant_storage=
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            # max_memory=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
            max_memory={0: "5GB"},  # Ajuste crítico
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        peft_config = LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
        )

        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, peft_config)

        self.training_args = TrainingArguments(
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=60,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            output_dir=self.output_dir,
            optim='adamw_8bit',
            weight_decay=0.01,
            lr_scheduler_type='linear',
            seed=3047
        )

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_args,
            train_dataset=load_dataset(
                path=self.datasets_path,
                data_files=['vehicle_bus_stop_occupation_20250101_20250201-copy.speaks.txt'],
                split="train"
            )
        )

    def train(self):
        self.trainer.train()
        self.model.save_pretrained(os.path.join(os.getcwd(), 'weight_models_sft'))
        self.tokenizer.save_pretrained(os.path.join(os.getcwd(), 'weight_models_sft'))

    
    def run(self):
        self.train()
    

    def load_trained_model(self, path):
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=50)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    


    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=50)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    


if __name__ == 'main':
    load_dotenv(override=True)
    argparser = argparse.ArgumentParser()
    argparse.add_argument('--mode', type=str, default=os.getenv('MODE', 'fine_tune'), choices=['fine_tune', 'predict'])


    # argparser.add_argument('--model_name', type=str, default=os.getenv('MODEL_NAME', 'unsloth/Meta-Llama-3.1-8B-bnb-4bit'))
    argparser.add_argument('--datasets_path', type=str, default=os.getenv('DATASETS_PATH', './data'))
    argparser.add_argument('--max_length', type=int, default=int(os.getenv('MAX_LENGTH', 4096)))
    argparser.add_argument('--batch_size', type=int, default=int(os.getenv('BATCH_SIZE', 2)))
    argparser.add_argument('--learning_rate', type=float, default=float(os.getenv('LEARNING_RATE', 2e-5)))
    argparser.add_argument('--num_epochs', type=int, default=int(os.getenv('NUM_EPOCHS', 3)))
    argparser.add_argument('--output_dir', type=str, default=os.getenv('OUTPUT_DIR', './results'))
    argparser.add_argument('--load_in_4bit', type=bool, default=os.getenv('LOAD_IN_4BIT', True))
    argparser.add_argument('--bnb_4bit_compute_dtype', type=str, default=os.getenv('BNB_4BIT_COMPUTE_DTYPE', 'float16'))
    argparser.add_argument('--bnb_4bit_quant_type', type=str, default=os.getenv('BNB_4BIT_QUANT_TYPE', 'nf4'))
    argparser.add_argument('--bnb_4bit_use_double_quant', type=bool, default=os.getenv('BNB_4BIT_USE_DOUBLE_QUANT', True))

    args = argparser.parse_args()

    if args.mode == 'fine_tune':
        bus_occupancy_fine_tune = BusOccupancyFineTune(
            model_name=args.model_name,
            datasets_path=args.datasets,
            max_length=args.max_length,
            batch_size=args.batch_size,
            learning_rate=args.learning,
            num_epochs=args.num_epochs,
            output_dir=args.output_dir,
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant
        )
        bus_occupancy_fine_tune.run()
    elif args.mode == 'predict':
        bus_occupancy_fine_tune = BusOccupancyFineTune()
        bus_occupancy_fine_tune.load_trained_model(path=args.output_dir)
        text = input("Enter your text: ")
        logger.debug(bus_occupancy_fine_tune.predict(text))
 




    