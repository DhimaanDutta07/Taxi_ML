CFG = {
    "postgres_url": "postgresql+psycopg2://postgres:123@localhost:5432/delivery_db",
    "raw_data": "data/raw/taxi_fare_synth_50k.csv",
    "cleaned_data": "data/processed/taxi_cleaned.csv",
    "featured_data": "data/processed/taxi_featured.csv",
    "random_state": 42,
    "test_size": 0.20,
    "target": "total_amount",

    "model_name": "xgb",
    "seed": 42,
    "cv": 3,
    "n_trials": 20,

    "model_path":"artifacts/model/model.pkl",
    "shap_report_path":"artifacts/shap/summary.png",

    "curr_path":"data/processed/taxi_cleaned.csv",
    "ref_path":"data/processed/taxi_cleaned.csv",
    "evidently_path":"artifacts/monitoring/evidently_report.html",
    "use_localstack": True,
    "s3_endpoint": "http://localhost:4566",
    "s3_bucket": "dvc-bucket",
}
