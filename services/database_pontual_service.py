import psycopg2
from psycopg2.extras import execute_batch
from typing import List
from models.weather_data import WeatherData
from datetime import date

class DatabaseService:
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.connect()


    def connect(self):
        self.connection = psycopg2.connect(**self.db_config)

    def list_vehiclerun_of_day(self, day: date):
        pass