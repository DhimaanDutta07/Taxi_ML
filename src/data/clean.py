import pandas as pd
from src.core.config import CFG

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        print("duplicated values:", self.df.duplicated().sum())
        print("nan values:\n", self.df.isna().sum())

    def clean(self):
        drop_cols = [
            "vendor_id", "tip_amount", "fare_amount", "weekend_surcharge",
            "tolls_amount", "night_surcharge",
            "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"
        ]

        self.df.drop(columns=drop_cols, inplace=True, errors="ignore")

        print("Dropped columns:", drop_cols)

        self.df.to_csv(CFG["cleaned_data"], index=False)
        print("Saved file to:", CFG["cleaned_data"])

        return self.df
