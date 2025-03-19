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
import pandas as pd
from datetime import datetime, timezone, timedelta
# from convert_data_to_json import convert_to_dict, convert_to_text

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)


days_of_week = {
    0: "domingo",
    1: "segunda",
    2: "terça",
    3: "quarta",
    4: "quinta",
    5: "sexta",
    6: "sabado"
}

TZ=timezone(timedelta(hours=-3))

def convert_tz(str_dt: str, tz: timezone):
    d = datetime.fromisoformat(str_dt)
    return d.astimezone(tz)

# human_speaks = "Sabendo que está %clima_tempo e a temperatura é %temperatura. Qual a lotação da linha %linha para o ponto %ponto para %dia às %hora?"
human_speaks = "Para uma viagem do itinerario %trip_route_id, que geralmente começa %trip_scheduled_time mas que começou %trip_start_time e terminou %trip_end_time, no dia %days_of_week, %weather_precipitation e temperatura %weather_temperature. Qual o nível de lotação para o ponto %bus_stop_id?"

bot_speaks = "No ponto %busStopId, a lotação é %occupancyLevel"


def convert_to_text(
        register: dict,
    ):
    logger.debug(f"Register(type): {type(register)}")
    logger.debug(f"Register: {register}")

    trip_route_id = register['tripRouteId']
    trip_scheduled_time = register['tripScheduledTime']
    trip_start_time = register['tripStartTime']
    trip_end_time = register['tripEndTime']        
    weather_precipitation = register['weatherPrecipitation']
    weather_temperature = register['weatherTemperature']
    bus_stop_id = register['busStopId']
    occupancy_level = register['occupancyLevel']
    normalized_location = register['busStopLocation']/register['routeTotalLength'],  


    scheduled_datetime = convert_tz(trip_scheduled_time, TZ)
    scheduled_time = scheduled_datetime.strftime("%H:%M")
    start_datetime = convert_tz(trip_start_time, TZ).strftime("%H:%M")
    end_time = convert_tz(trip_end_time, TZ).strftime("%H:%M")
    
    dw = days_of_week[scheduled_datetime.weekday()]

    occupancy = 'Lotado' if occupancy_level > 1 else 'Vazio' if occupancy_level < 1 else 'Cheio'



                        



    human_speaking = human_speaks\
        .replace("%trip_route_id", str(trip_route_id))\
        .replace("%trip_scheduled_time", scheduled_time)\
        .replace("%trip_start_time", start_datetime)\
        .replace("%trip_end_time", end_time)\
        .replace("%days_of_week", dw)\
        .replace("%weather_precipitation", 'Chovendo' if weather_precipitation > 0 else 'Não Chovendo')\
        .replace("%weather_temperature", 'Quente' if weather_temperature > 20 else 'Frio')\
        .replace("%bus_stop_id", str(bus_stop_id))

    bot_speaking = bot_speaks\
        .replace("%busStopId", str(bus_stop_id))\
        .replace("%occupancyLevel", occupancy)

  
    return f"""<human>: {human_speaking}\n<bot>:{bot_speaking}\n\n"""



    

class BusOccupancyFineTune:
    """
    This is a fine tune class to manage the fine tune process for bus occupancy data.
    """

   
    def __init__(self, mode: str = "test_dataset",  **kwargs):
        
        self.model_name = kwargs.get("model_name", "bert-base-uncased")
        output_default_dir = f'tuned_{self.model_name}'
        self.datasets_path = kwargs.get("datasets_path", "./data")
        self.output_dir = kwargs.get("output_dir", output_default_dir)
        self.data_batch_size = kwargs.get("data_batch_size", 1000)  # New parameter for data batch size
        
        if mode == 'predict':
            self.load_trained_model()

        elif mode == 'fine_tune':
            
            self.max_length = kwargs.get("max_length", 128)
            self.batch_size = kwargs.get("batch_size", 16)
            self.learning_rate = kwargs.get("learning_rate", 2e-5)
            self.num_epochs = kwargs.get("num_epochs", 3) 
            load_in_4bit = kwargs.get("load_in_4bit", False)
            bnb_4bit_compute_dtype = kwargs.get("bnb_4bit_compute_dtype", "float16")
            bnb_4bit_quant_type = kwargs.get("bnb_4bit_quant_type", "nf4")
            bnb_4bit_use_double_quant = kwargs.get("bnb_4bit_use_double_quant", True)

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                # max_memory=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
                max_memory={0: "3.5GB"},  # Critical adjustment
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

    
    def train(self):
        for batch in self.read_datasets_in_batches():
            self.trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                args=self.training_args,
                train_dataset=batch
            )
            self.trainer.train()
        

        self.model.save_pretrained(os.path.join(os.getcwd(), self.output_dir))
        self.tokenizer.save_pretrained(os.path.join(os.getcwd(), self.output_dir))

    def run(self):
        self.train()

    def load_trained_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=self.output_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.output_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=50)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def read_datasets_in_batches(self):
        logger.debug(f"Reading datasets in batches: {self.datasets_path}")
        filenames = sorted(os.listdir(self.datasets_path))
        logger.debug(f"filenames: {filenames}")
        for f in filenames:
            if f.endswith('.jsonl'):
                logger.debug(f"Processing file: {f}")
                file_path = os.path.join(self.datasets_path, f)
                if os.path.isfile(file_path):
                    df = pd.read_json(file_path, lines=True)
                    acc = []
                    for index, row in df.iterrows():
                        acc.append(convert_to_text(row))
                        if len(acc) == self.data_batch_size:
                            yield acc
                            acc = []
                    



                    # for start in range(0, len(df), self.data_batch_size):
                    #     # batch_to_train = df[start:start + self.data_batch_size]
                    #     batch_to_train = df.loc[start:start + self.data_batch_size]

                    #     logger.debug(f"Batch type: {type(batch_to_train['tripScheduledTime'])}")

                    #     l = [convert_to_text(r) for r in  batch_to_train]

                    #     yield l


                        

                        
                        # yield batch_to_train
                        # logger.trace(f"Batch: \n{batch_to_train[0:10]}")
                        
                        # text = convert_to_text(
                        #     batch_to_train['tripRouteId'],
                        #     batch_to_train['busStopId'],
                        #     batch_to_train['readingTime'],
                        #     batch_to_train['occupation'],
                        #     batch_to_train['normalizedLocation'],
                        #     batch_to_train['humidity'],
                        #     batch_to_train['temperature']
                        # )
                        # yield text




if __name__ == '__main__':
    load_dotenv(override=True)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mode', type=str, default=os.getenv('MODE', 'fine_tune'), choices=['fine_tune', 'predict', 'test_dataset'])
    argparser.add_argument('--model_name', type=str, default=os.getenv('MODEL_NAME', 'unsloth/Meta-Llama-3.1-8B-bnb-4bit'))
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
    argparser.add_argument('--data_batch_size', type=int, default=int(os.getenv('DATA_BATCH_SIZE', 1000)))  # New argument

    args = argparser.parse_args()

    logger.info(f"Arguments: {args}")
    

    if args.mode == 'test_dataset':
        bus_occupancy_fine_tune = BusOccupancyFineTune(
            datasets_path=args.datasets_path,
            data_batch_size=args.data_batch_size
            )
        # logger.debug(f"Reading datasets in batches: {datasets_path}")
        l = bus_occupancy_fine_tune.read_datasets_in_batches()
        resp = next(l)
        print('\n'.join(resp))
        l.close()
        


    if args.mode == 'fine_tune':
        bus_occupancy_fine_tune = BusOccupancyFineTune(
            mode='fine_tune',
            model_name=args.model_name,
            datasets_path=args.datasets_path,
            max_length=args.max_length,            
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            output_dir=args.output_dir,
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            data_batch_size=args.data_batch_size  # Pass the new argument
        )
        bus_occupancy_fine_tune.run()
    elif args.mode == 'predict':
        bus_occupancy_fine_tune = BusOccupancyFineTune(mode='predict')
        bus_occupancy_fine_tune.load_trained_model(path=args.output_dir)
        text = input("Enter your text: ")
        logger.debug(bus_occupancy_fine_tune.predict(text))
