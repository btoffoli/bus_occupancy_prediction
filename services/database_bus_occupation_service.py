# services/database_service.py
import psycopg2
from psycopg2.extras import execute_batch, execute_values
from typing import List
from models.pontual_data import VehiclerunBusStopoccupation
import logging
from .file_reader import FileReader
from datetime import datetime
import pandas as pd
import os
from pathlib import Path
import sys
logger = logging.getLogger(__name__)


class DatabaseService:
    def __init__(self, file_path: str, batch_size: int = 1000):
        self.batch_size = batch_size
        self.file_path = file_path
    
    def insert_vehiclerun_busstop_occupation_batch(self, data: List[VehiclerunBusStopoccupation]):
        new_df = pd.DataFrame((wd.to_dict() for wd in data))

        # Adicionando os novos dados ao arquivo CSV existente
        # print_header = not os.path.isfile(self.file_path)
        print_header = not os.path.exists(self.file_path)
        # print(f"print_header: {print_header} - modo: {'w' if print_header else 'a'}")
        # não funcionou pois o controle do flush não é meu, dessa forma, se na proxima chamada o header estiver false ele não imprime o cabeçalho, e é o que está acontecendo
        # new_df.to_csv(self.file_path, mode='a', header= print_header, index=False)
        with open(self.file_path, 'a') as f:
            if print_header:
                f.write(','.join(list(data[0].to_dict().keys())))
                f.write('\n')
                f.flush()
            new_df.to_csv(f, header=False, index=False)
            
                

        
        