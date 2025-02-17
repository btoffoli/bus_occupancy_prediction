import psycopg2
from psycopg2.extras import execute_batch
from typing import List
from models.weather_data import WeatherData
from datetime import date, timedelta


class DatabaseService:
    def __init__(self, 
                 db_config: dict, 
                 start_date: date, 
                 end_date: date, 
                 batch_size: int = 1000):        
        if not start_date:
            raise ValueError("start_date is required")
        if not end_date:
            raise ValueError("end_date is required")
        if end_date <= start_date:
            raise ValueError("end_date must be greater than start_date")        self.start_date = start_date
        self.start_date = start_date            
        self.end_date = end_date        
        self.batch_size = batch_size
        self.db_config = db_config

        self.current_date = start_date

        self.connect()

    


    def connect(self):
        self.connection = psycopg2.connect(**self.db_config)

    def list_vehiclerun_of_day(self, day: date):
        with self.connection.cursor() as cursor:
            cursor.execute(sql_list_vehiclerun_of_day, (day, day + timedelta(days=1)))
            return cursor.fetchall()
        

    def run(self):
        while self.current_date <= self.end_date:
            vehicleruns = self.list_vehiclerun_of_day(self.current_date)
            self.current_date += timedelta(days=1)
            print(f"vehicleruns: {vehicleruns}")
        



sql_list_vehiclerun_of_day = """
    SELECT
	vr.forwarditinerary_oid itinerary_id,
	vr.scheduledtime,
	vr.tripstarttime,
	vr.tripcompletiontime,
	vr.vehicletranscol_oid vehicle_id
FROM
	transcol.vehiclerun_202501 vr
WHERE
	TRUE	
	AND vr.scheduledtime BETWEEN %s::DATE AND %s::DATE
	AND vr.status = 5
"""