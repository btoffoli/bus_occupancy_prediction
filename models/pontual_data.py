# models/weather_data.py
from dataclasses import dataclass
from datetime import datetime

@dataclass
class VehicleRunData:
    itinerary_id: int
    scheduledtime: datetime
    tripstarttime: datetime
    tripcompletiontime: datetime
    vehicle_id: int
    
    