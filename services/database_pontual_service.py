import psycopg2
from psycopg2.extras import execute_batch
from typing import List
from models.weather_data import WeatherData
from models.pontual_data import VehicleRunData, Itinerary, ItineraryBusstopAssociation, BusOccupation
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
            raise ValueError("end_date must be greater than start_date")        
        self.start_date = start_date            
        self.end_date = end_date        
        self.batch_size = batch_size
        self.db_config = db_config

        self.current_date = start_date

        self.itinerarys : dict[int, Itinerary] = dict()
        self.bus_ocuppation : list[BusOccupation] = []


        self.connect()

    


    def connect(self):
        self.connection = psycopg2.connect(**self.db_config)

    def list_vehiclerun_of_day(self, day: date):
        with self.connection.cursor() as cursor:
            cursor.execute(sql_list_vehiclerun_of_day.replace('YEARMONTH', day.strftime('%Y%m')), (day, day + timedelta(days=1)))
            return cursor.fetchall()
        
    def load_itinerarys_not_loaded(self, itineray_ids: set[int]):
        # itineray_ids_must_be_loaded = self.itinerarys.keys() - itineray_ids
        itineray_ids_loaded = set(self.itinerarys.keys())
        print(f"itineray_ids_loaded: {len(itineray_ids_loaded)}")
        itineray_ids.difference_update(itineray_ids_loaded)
        print(f"itineray_ids: {len(itineray_ids)}")
        print(f"itineray_ids: {list(itineray_ids)[:10]}")

        if not itineray_ids:
            print("itineray_ids is empty")
            return

        with self.connection.cursor() as cursor:
            cursor.execute(sql_list_itinerarys, (list(itineray_ids),))
            l = cursor.fetchall()
            for i in l:
                self.itinerarys[int(i[0])] = Itinerary(*i)

        # load ItineraryBusstopAssociation
        with self.connection.cursor() as cursor:
            cursor.execute(sql_list_itinerarys_busstop_associations, (list(itineray_ids),))
            l = cursor.fetchall()            
            itinerarys_busstop_association_current = None
            for i in l:
                itinerarys_busstop_association = ItineraryBusstopAssociation(*i)
                itinerary = self.itinerarys.get(itinerarys_busstop_association.itinerary_id)
                itinerary.itinerary_busstop_associations.append(itinerarys_busstop_association)


        print(f"self.itinerarys: {len(self.itinerarys)}\n\n\n\n")

    def load_bus_occupation_of_day(self, day: date, vehicle_ids: set[int]):
        if not vehicle_ids:
            return
        self.bus_ocuppation.clear()
        with self.connection.cursor() as cursor:
            cursor.execute(sql_list_bus_occupation_of_day.replace('YEARMONTH', day.strftime('%Y%m')), (list(set(id for id in vehicle_ids)), day, day + timedelta(days=1)))
            l = cursor.fetchall()            
            for i in l:
                self.bus_ocuppation.append(BusOccupation(*i))

        print(f"self.bus_ocuppation: {len(self.bus_ocuppation)}")
                


        

    def run(self):
        while self.current_date <= self.end_date:
            print(f"Processing current_date: {self.current_date}")
            vehicleruns = [VehicleRunData(*vr) for vr in self.list_vehiclerun_of_day(self.current_date)]
            print(f"vehicleruns: {len(vehicleruns)}")
            itinerarys = set(vr.itinerary_id for vr in vehicleruns)
            self.load_itinerarys_not_loaded(itinerarys)
            self.load_bus_occupation_of_day(self.current_date, set(vr.vehicle_id for vr in vehicleruns))
            # print(f"itinerarys: {len(self.itinerarys)}")
            # print(f"vehicleruns: {vehicleruns}")
            self.current_date += timedelta(days=1)
            
            

    def close(self):
        if self.connection and self.connection.closed == 0:
            self.connection.close()
    



sql_list_vehiclerun_of_day = """
    SELECT
	vr.forwarditinerary_oid itinerary_id,
	vr.scheduledtime,
	vr.tripstarttime,
	vr.tripcompletiontime,
	vr.vehicletranscol_oid vehicle_id
FROM
	transcol.vehiclerun_YEARMONTH vr
WHERE
	TRUE	
	AND vr.scheduledtime BETWEEN %s::DATE AND %s::DATE
	AND vr.status = 5
"""

sql_list_itinerarys = """
    SELECT
        id,
        codigo as code,
        --ST_AsEWKB(geometria) as geo
        ST_AsText(geometria) as geo
    FROM
        pontual.itinerario     
    WHERE id = ANY(%s::INT[]) 
"""

sql_list_itinerarys_busstop_associations = """
    SELECT
        api.id,
        api.itinerario_id,
        p.id busstop_id,
        p.codigo bustop_code,
        api.localizacao "location"       
    FROM
        pontual.associacao_ponto_itinerario api
    JOIN
        pontual.ponto_de_parada p
            ON p.id = api.ponto_de_parada_id
    WHERE api.itinerario_id = ANY(%s::INT[])
    ORDER BY itinerario_id, "location"
"""

sql_list_bus_occupation_of_day = """
    SELECT
        id,
        veiculo_id vehicle_id,
        horario reading_time,
        nivel_lotacao occupation,
        ST_AsText(localizacao) geo
    FROM
        pontual.lotacao_veiculo_YEARMONTH
    WHERE
        TRUE
        AND veiculo_id = ANY(%s::INT[])
        AND horario BETWEEN %s::DATE AND %s::DATE
        AND localizacao IS NOT NULL
    
    ORDER BY
        veiculo_id,
        horario
"""


