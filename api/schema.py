from pydantic import BaseModel


class TripRequest(BaseModel):
    pickup_datetime: str
    passenger_count: int
    rate_code: str
    payment_type: str
    trip_distance_km: float
    trip_duration_min: float
    total_amount: float
    pickup_hour: int
    car_type: str
    is_airport: int
    is_outer_borough: int
    distance_type: str
    trip_duration_hours: float
    duration_type: str
    is_long_trip: int
    avg_speed_kmh: float
    is_night_time: int


class PredictResponse(BaseModel):
    prediction: float
