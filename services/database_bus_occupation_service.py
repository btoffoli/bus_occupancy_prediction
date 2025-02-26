# services/database_service.py
import psycopg2
from psycopg2.extras import execute_batch
from typing import List
from models.pontual_data import VehiclerunBusStopoccupation
import logging
from .file_reader import FileReader

logger = logging.getLogger(__name__)


class DatabaseService:
    def __init__(self, db_config: dict, batch_size: int = 1000):
        self.db_config = db_config        
        self.batch_size = batch_size
        self.connect()
        

    def connect(self):
        self.connection = psycopg2.connect(**self.db_config)
        

    def insert_vehiclerun_busstop_occupation_batch(self, vechicle_busstop_occupations: List[VehiclerunBusStopoccupation]):
        print(f"vechicle_busstop_occupations: {vechicle_busstop_occupations}")


        try:
            query = """
            INSERT INTO public.vehiclerunbusstopoccupation (
                itinerary_code,
                itinerary_size,
                scheduled_time,
                trip_start_time,
                trip_completion_time,
                vehicle_id,
                busstop_code,
                bustop_location,
                reading_time,
                occupation,
                occupation_geo,
                occupation_location,
                normalized_location,
                temperature,
                humidity,
                wind_speed,
                precipitation

            )
            VALUES (
                %(itinerary_code)s,
                %(itinerary_size)s,
                %(scheduled_time)s,
                %(trip_start_time)s,
                %(trip_completion_time)s,
                %(vehicle_id)s,
                %(busstop_code)s,
                %(bustop_location)s,
                %(reading_time)s,
                %(occupation)s,
                %(occupation_geo)s,
                %(occupation_location)s,
                %(normalized_location)s,
                %(temperature)s,
                %(humidity)s,
                %(wind_speed)s,
                %(precipitation)s
            )
            ON CONFLICT (itinerary_code, scheduled_time, busstop_code)
            DO UPDATE SET
                itinerary_size = EXCLUDED.itinerary_size,
                trip_start_time = EXCLUDED.trip_start_time,
                trip_completion_time = EXCLUDED.trip_completion_time,
                vehicle_id = EXCLUDED.vehicle_id,
                bustop_location = EXCLUDED.bustop_location,
                reading_time = EXCLUDED.reading_time,
                occupation = EXCLUDED.occupation,
                occupation_geo = EXCLUDED.occupation_geo,
                occupation_location = EXCLUDED.occupation_location,
                normalized_location = EXCLUDED.normalized_location;
                temperature = EXCLUDED.temperature,
                humidity = EXCLUDED.humidity,
                wind_speed = EXCLUDED.wind_speed,
                precipitation = EXCLUDED.precipitation;
            """
            data = [
                wd.to_dict()
                for wd in vechicle_busstop_occupations
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


    def reinitialize(self):
        self.close()
        self.connect()
        

    def close(self):
        if self.connection and self.connection.closed == 0:
            self.connection.close()
            self.connection = None
            

   