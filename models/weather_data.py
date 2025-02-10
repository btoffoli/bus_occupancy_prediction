# models/weather_data.py
from dataclasses import dataclass
from datetime import datetime

@dataclass
class WeatherData:
    city: str
    reading_time: datetime
    precipitation: float
    station_pressure: float
    max_pressure: float
    min_pressure: float
    global_radiation: float
    temperature: float
    dew_point: float
    max_temperature: float
    min_temperature: float
    max_dew_point: float
    min_dew_point: float
    max_humidity: float
    min_humidity: float
    humidity: float
    wind_direction: float
    wind_gust: float
    wind_speed: float