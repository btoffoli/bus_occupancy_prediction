import psycopg2
from psycopg2.extras import execute_batch
from typing import List
from models.weather_data import WeatherData
from models.pontual_data import VehicleRunData, Itinerary, ItineraryBusstopAssociation, BusOccupation, VehiclerunBusStopoccupation
from datetime import date, timedelta, datetime
from .database_bus_occupation_service import DatabaseService as DatabaseBusOccupationService
from .database_weather_service import DatabaseService as DatabaseWeatherService
import logging
from pyproj import Transformer
from shapely.geometry import LineString, Point
from .utils import read_last_line





logger = logging.getLogger(__name__)

# Define the coordinate transformation from WGS84 to UTM zone 24S, 
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32724", always_xy=True)

class DatabaseService:

    def __init__(self, 
                 db_config: dict, 
                 start_date: date, 
                 end_date: date,
                 database_weather_data_service: DatabaseWeatherService,
                 database_bus_occupation_service: DatabaseBusOccupationService,
                 file_path: str,                 
                 batch_size: int = 1000):        
        if not start_date:
            raise ValueError("start_date is required")
        if not end_date:
            raise ValueError("end_date is required")
        if end_date < start_date:
            raise ValueError("end_date must be greater than start_date")        
        self.start_date = start_date            
        self.end_date = end_date        
        self.batch_size = batch_size
        self.db_config = db_config
        self.database_bus_occupation_service = database_bus_occupation_service
        self.database_weather_data_service = database_weather_data_service
        self.database_weather_data_service.load_data()

        last_line = read_last_line(file_path)
        if last_line is not None:
            self.current_scheduled_time =  datetime.fromisoformat(last_line['scheduled_time'])
            self.current_itinerary_code = last_line['itinerary_code']
            self.current_date = self.current_scheduled_time.date()
        else:
            self.current_date = start_date
            self.current_scheduled_time = None
            self.current_itinerary_code = None
        
        self.__initialize()
        self.connect()


    def __initialize(self):
        self.itinerarys : dict[int, Itinerary] = dict()
        self.bus_occupation : list[BusOccupation] = []
        self.vehiclerun_bus_stop_ocuppations = []

    
    def __reinitialize(self):
        self.__initialize()        
        self.close()
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
        logger.debug(f"itineray_ids_loaded: {len(itineray_ids_loaded)}")
        itineray_ids.difference_update(itineray_ids_loaded)
        logger.debug(f"itineray_ids: {len(itineray_ids)}")
        logger.debug(f"itineray_ids: {list(itineray_ids)[:10]}")

        if not itineray_ids:
            logger.debug("itineray_ids is empty")
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
                itinerary.add_itinerary_busstop_association(itinerarys_busstop_association)


        logger.debug(f"self.itinerarys: {len(self.itinerarys)}\n\n\n\n")

    def load_bus_occupation_of_day(self, day: date, vehicle_ids: set[int]):
        if not vehicle_ids:
            return
        self.bus_occupation.clear()
        with self.connection.cursor() as cursor:
            cursor.execute(sql_list_bus_occupation_of_day.replace('YEARMONTH', day.strftime('%Y%m')), (list(set(id for id in vehicle_ids)), day, day + timedelta(days=1)))
            l = cursor.fetchall()            
            for i in l:
                self.bus_occupation.append(BusOccupation(*i))

        logger.debug(f"self.bus_occupation: {len(self.bus_occupation)}")

    
    def load_vehiclerun_bus_stop_ocuppation(self, vehiclerun: VehicleRunData):
        resp = []
        # for vehiclerun in self.vehicleruns:
        itinerary = self.itinerarys.get(vehiclerun.itinerary_id)
        if not itinerary:
            logger.debug(f"itinerary not found for vehiclerun: {vehiclerun}")
            return

        if not itinerary.itinerary_busstop_associations:
            logger.debug(f"itinerary_busstop_associations not found for itinerary: {itinerary}")
            return
        
        transformed_itinerary = LineString([transformer.transform(x, y) for x, y in itinerary.geo.coords])
        reversed_bus_occupation = sorted(filter(lambda x: x.vehicle_id == vehiclerun.vehicle_id and x.reading_time >= vehiclerun.tripstarttime and x.reading_time < vehiclerun.tripcompletiontime, self.bus_occupation), key=lambda x: x.reading_time, reverse=True)
        for association in itinerary.itinerary_busstop_associations:
            bus_stop_location = association.location

            # find last bus ocuppation before bus stop in current association
            last_bus_occupation, occupation_location = None, None
            for bus_occupation in reversed_bus_occupation:
                transformed_bus_occupation_point = Point([transformer.transform(x, y) for x, y in bus_occupation.geo.coords])
                occupation_location = transformed_itinerary.project(transformed_bus_occupation_point)                
                if occupation_location < bus_stop_location:
                    last_bus_occupation = bus_occupation
                    break

            
            if last_bus_occupation is not None:
                vehiclerun_bus_stop_ocuppation = VehiclerunBusStopoccupation(
                        itinerary,
                        vehiclerun,
                        last_bus_occupation,
                        association,
                        occupation_location
                    )
                
                weather_data = self.database_weather_data_service.get_weather_data(vehiclerun_bus_stop_ocuppation.reading_time)
                if weather_data:
                    vehiclerun_bus_stop_ocuppation.temperature = weather_data.temperature
                    vehiclerun_bus_stop_ocuppation.humidity = weather_data.humidity
                    vehiclerun_bus_stop_ocuppation.wind_speed = weather_data.wind_speed
                    vehiclerun_bus_stop_ocuppation.precipitation = weather_data.precipitation
            
                logger.debug(f"vehiclerun_bus_stop_ocuppation: {vehiclerun_bus_stop_ocuppation}")

                resp.append(vehiclerun_bus_stop_ocuppation)

            self.vehiclerun_bus_stop_ocuppations.extend(resp)

        return resp

        

    def run(self):
        while self.current_date <= self.end_date:
            logger.debug(f"Processing current_date: {self.current_date}")
            vehicleruns = [VehicleRunData(*vr) for vr in self.list_vehiclerun_of_day(self.current_date)]
            logger.debug(f"vehicleruns: {len(vehicleruns)}")
            sorted_vehicleruns = sorted(vehicleruns, key=lambda x: (x.itinerary_code, x.scheduledtime))
            print(f"sorted_vehicleruns: {[i.itinerary_code for i in sorted_vehicleruns[:10]]}")
           
            itinerarys = set(vr.itinerary_id for vr in sorted_vehicleruns)
            self.load_itinerarys_not_loaded(itinerarys)
            self.load_bus_occupation_of_day(self.current_date, set(vr.vehicle_id for vr in vehicleruns))
            for vehiclerun in vehicleruns:
                
                if self.current_scheduled_time and self.current_itinerary_code:
                    vechiclerun_itinerary_code = self.itinerarys[vehiclerun.itinerary_id].code
                    if vechiclerun_itinerary_code <= self.current_itinerary_code\
                        and vehiclerun.scheduledtime <= self.current_scheduled_time:
                        print(f"Pulando itinerario {vechiclerun_itinerary_code} - agendado: {vehiclerun.scheduledtime} - atual: {self.current_scheduled_time}")
                        continue
                
                partial_vehiclerun_bus_stop_ocuppation = self.load_vehiclerun_bus_stop_ocuppation(vehiclerun)
                if partial_vehiclerun_bus_stop_ocuppation:
                    self.database_bus_occupation_service.insert_vehiclerun_busstop_occupation_batch(partial_vehiclerun_bus_stop_ocuppation)
                    self.vehiclerun_bus_stop_ocuppations.extend(partial_vehiclerun_bus_stop_ocuppation)
            
            logger.debug(f"self.vehiclerun_bus_stop_ocuppations: {len(self.vehiclerun_bus_stop_ocuppations)}")
            
            # self.database_bus_occupation_service.insert_vehiclerun_busstop_occupation_batch(self.vehiclerun_bus_stop_ocuppations, self.batch_size)            


            self.__reinitialize()
            # logger.debug(f"itinerarys: {len(self.itinerarys)}")
            # logger.debug(f"vehicleruns: {vehicleruns}")
            self.current_date += timedelta(days=1)
            
            

    def close(self):
        if self.connection and self.connection.closed == 0:
            self.connection.close()
    



sql_list_vehiclerun_of_day = """
    SELECT
	vr.forwarditinerary_oid itinerary_id,
    i.codigo itinerary_code,
	vr.scheduledtime,
	vr.tripstarttime,
	vr.tripcompletiontime,
	vr.vehicletranscol_oid vehicle_id
FROM
	transcol.vehiclerun_YEARMONTH vr
    JOIN pontual.itinerario i
        ON i.id = vr.forwarditinerary_oid
WHERE
	TRUE	
	AND vr.scheduledtime BETWEEN %s::DATE AND %s::DATE
    AND vr.tripstarttime IS NOT NULL
    AND vr.tripcompletiontime IS NOT NULL
	AND vr.status = 5
    AND vr.itineraryobservance = 'FULL'
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
        api.itinerario_id itinerary_id,
        p.id busstop_id,
        p.codigo busstop_code,        
        api.localizacao "location",
        ST_ASText(p.geometria) busstop_location
    FROM
        pontual.associacao_ponto_itinerario api
    JOIN
        pontual.ponto_de_parada p
            ON p.id = api.ponto_de_parada_id
    WHERE 
        TRUE
        AND api.itinerario_id = ANY(%s::INT[])
        AND api.localizacao IS NOT NULL
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


