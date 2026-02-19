import pandas as pd
from src.core.config import CFG

class FeatureBuilder():
    def build(self,df):
        df=df.copy()
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
        df["pickup_hour"] = df["pickup_datetime"].dt.hour
        df["car_type"]=df["passenger_count"].apply(lambda x:"4_seater" if x<=4 else "6_seater")
        df["is_airport"]=df["rate_code"].apply(lambda x: 1 if x=="airport" else 0)
        df["is_outer_borough"]=df["rate_code"].apply(lambda x: 1 if x=="outer_borough" else 0)
        avg_dist=15
        df["distance_type"]=df["trip_distance_km"].apply(lambda x: "short" if x<=avg_dist else "long")
        df["trip_duration_hours"]=df["trip_duration_min"]/60
        avg_time=0.80
        df["duration_type"]=df["trip_duration_hours"].apply(lambda x: "long" if x>avg_time else "short")
        df["is_long_trip"] = ((df["duration_type"] == "long") & (df["distance_type"] == "long")).astype(int)
        df["avg_speed_kmh"] = df["trip_distance_km"] / df["trip_duration_hours"]
        df["is_night_time"] = ((df["pickup_hour"] >= 20) | (df["pickup_hour"] <= 4)).astype(int)
        df.to_csv(CFG["featured_data"], index=False)
        print("featured_cols")
        print(df[["pickup_hour","car_type","is_airport","is_outer_borough","distance_type","trip_duration_hours","duration_type","is_long_trip","avg_speed_kmh"]])
        return df



    