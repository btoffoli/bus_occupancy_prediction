# services/database_service.py
import psycopg2
from psycopg2.extras import execute_batch
from typing import List
from models.weather_data import WeatherData
import logging
from .file_reader import FileReader

logger = logging.getLogger(__name__)


class DatabaseService:
    def __init__(self, db_config: dict, data_dir: str, batch_size: int = 1000):
        self.db_config = db_config
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.connect()
        

    def connect(self):
        self.connection = psycopg2.connect(**self.db_config)
        

    def insert_weather_data_batch(self, weather_data_list: List[WeatherData]):
        # print(f"weather_data_list: {weather_data_list}")
        try:
            query = """
            INSERT INTO weather_data (
                city, reading_time, precipitation, station_pressure, max_pressure, min_pressure,
                global_radiation, temperature, dew_point, max_temperature, min_temperature,
                max_dew_point, min_dew_point, max_humidity, min_humidity, humidity,
                wind_direction, wind_gust, wind_speed
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (city, reading_time) 
            DO UPDATE SET
                precipitation = EXCLUDED.precipitation,
                station_pressure = EXCLUDED.station_pressure,
                max_pressure = EXCLUDED.max_pressure,
                min_pressure = EXCLUDED.min_pressure,
                global_radiation = EXCLUDED.global_radiation,
                temperature = EXCLUDED.temperature,
                dew_point = EXCLUDED.dew_point,
                max_temperature = EXCLUDED.max_temperature,
                min_temperature = EXCLUDED.min_temperature,
                max_dew_point = EXCLUDED.max_dew_point,
                min_dew_point = EXCLUDED.min_dew_point,
                max_humidity = EXCLUDED.max_humidity,
                min_humidity = EXCLUDED.min_humidity,
                humidity = EXCLUDED.humidity,
                wind_direction = EXCLUDED.wind_direction,
                wind_gust = EXCLUDED.wind_gust,
                wind_speed = EXCLUDED.wind_speed;
            """
            data = [
                (
                    wd.city, wd.reading_time, wd.precipitation, wd.station_pressure, wd.max_pressure, wd.min_pressure,
                    wd.global_radiation, wd.temperature, wd.dew_point, wd.max_temperature, wd.min_temperature,
                    wd.max_dew_point, wd.min_dew_point, wd.max_humidity, wd.min_humidity, wd.humidity,
                    wd.wind_direction, wd.wind_gust, wd.wind_speed
                )
                for wd in weather_data_list
            ]

            print(f"data: {data}")
            if not self.connection or self.connection.closed == 1:
                self.connect()
                
            
            with self.connection.cursor() as cursor:
                # self.connection.tpc_begin()
                execute_batch(cursor, query, data)
        except Exception as e:
            self.connection.rollback()
            print(f"Error inserting data: {e}")
            raise
        else:
            self.connection.commit()


    def destroy(self):
        self.connection.close()
    def close(self):
        if self.connection:
            self.connection.close()    

    
    def run(self):
        # Processa cada arquivo CSV no diretório
        try:        
            for csv_file in self.data_dir.glob('*.CSV'):
                logger.info(f"Processing file: {csv_file}")
                try:
                    for weather_data_list in FileReader.read_csv(csv_file):
                        logger.info(f"weather_data: {len(weather_data_list)}")
                        self.insert_weather_data_batch(weather_data_list)
                    logger.info(f"Successfully processed file: {csv_file}")
                except Exception as e:
                    logger.error(f"Error processing file {csv_file}: {e}")
                
                finally:
                    self.connection.commit()
        except Exception as e:
            logger.error(f"Error processing data directory: {e}")
            raise e
        finally:
            self.close()



                

