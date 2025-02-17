# models/weather_data.py
from dataclasses import dataclass
from datetime import datetime
from shapely import LineString, Point, wkb, from_wkt

@dataclass
class VehicleRunData:
    itinerary_id: int
    scheduledtime: datetime
    tripstarttime: datetime
    tripcompletiontime: datetime
    vehicle_id: int

    def __init__(self, itinerary_id, scheduledtime, tripstarttime, tripcompletiontime, vehicle_id):
        self.itinerary_id = itinerary_id
        self.scheduledtime = scheduledtime
        self.tripstarttime = tripstarttime
        self.tripcompletiontime = tripcompletiontime
        self.vehicle_id = vehicle_id
    

    
@dataclass
class BusStop:
    id: int
    code: str
    geo: Point


@dataclass
class BusOccupation:
    id: int
    vehicle_id: int
    reading_time: datetime
    occupation: int
    geo: Point


    def __init__(self,
            id: int,
            vehicle_id: int,
            reading_time: datetime,
            occupation: int,
            geo: str
            
        ):
        self.vehicle_id = vehicle_id
        self.reading_time = reading_time
        self.occupation = occupation
        self.geo = from_wkt(geo)






@dataclass
class ItineraryBusstopAssociation:
    id: int
    itinerary_id: int
    busstop_id: int
    bustop_code: str
    location: int

    def __init__(self, id, itinerary_id, busstop_id, bustop_code, location):
        self.id = id
        self.itinerary_id = itinerary_id
        self.busstop_id = busstop_id
        self.bustop_code = bustop_code
        self.location = location

    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return self.id == other.id




@dataclass
class Itinerary:
    id: int
    code: str
    geo: LineString

    def __init__(self, id: int, code : str, geo: str):
        # print(f"geometry: {geo}")
        # print(f"type(geometry): {type(geo)}")
        # if isinstance(geometry, memoryview):
        #     geometry = bytes(geometry)
        self.id = id
        self.code = code
        # self.geo = wkb.load(geo, hex=True)
        self.geo = from_wkt(geo)
        # print(f"self.geo: {type(self.geo)}")
        self.itinerary_busstop_associations : list[ItineraryBusstopAssociation] = []


    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id
    




    
    