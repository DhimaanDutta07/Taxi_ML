import os
from fastapi import APIRouter, HTTPException
from api.schema import TripRequest, PredictResponse

router = APIRouter()

MODEL_PATH = os.path.join("artifacts", "model", "model.pkl")
_model = None


def load_model_anyhow(path: str):
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        pass

    try:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass

    try:
        import pickle
        with open(path, "rb") as f:
            return pickle.Unpickler(f, encoding="latin1").load()
    except Exception:
        pass

    try:
        import cloudpickle
        with open(path, "rb") as f:
            return cloudpickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Model load failed: {e}")


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model file not found at: {MODEL_PATH}")
        _model = load_model_anyhow(MODEL_PATH)
    return _model


@router.post("/predict", response_model=PredictResponse)
def predict(req: TripRequest):
    try:
        import pandas as pd

        model = get_model()

        row = {
            "pickup_datetime": req.pickup_datetime,
            "passenger_count": req.passenger_count,
            "rate_code": req.rate_code,
            "payment_type": req.payment_type,
            "trip_distance_km": req.trip_distance_km,
            "trip_duration_min": req.trip_duration_min,
            "total_amount": req.total_amount,
            "pickup_hour": req.pickup_hour,
            "car_type": req.car_type,
            "is_airport": req.is_airport,
            "is_outer_borough": req.is_outer_borough,
            "distance_type": req.distance_type,
            "trip_duration_hours": req.trip_duration_hours,
            "duration_type": req.duration_type,
            "is_long_trip": req.is_long_trip,
            "avg_speed_kmh": req.avg_speed_kmh,
            "is_night_time": req.is_night_time,
        }

        X = pd.DataFrame([row])
        y = model.predict(X)

        return PredictResponse(prediction=float(y[0]))

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")
