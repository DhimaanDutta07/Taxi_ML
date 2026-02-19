import pandas as pd
import boto3
from botocore.config import Config
from sqlalchemy import create_engine
from src.core.config import CFG


class DataLoader:
    def __init__(self, conn_url=None, table=None):
        self.conn_url = conn_url or CFG["postgres_url"]
        self.table = table
        self.engine = create_engine(self.conn_url)

        if CFG.get("use_localstack", False):
            self._ensure_localstack_s3()

    def _ensure_localstack_s3(self):
        s3 = boto3.client(
            "s3",
            endpoint_url=CFG.get("s3_endpoint", "http://localhost:4566"),
            aws_access_key_id="test",
            aws_secret_access_key="test",
            region_name="us-east-1",
            config=Config(s3={"addressing_style": "path"}),
        )

        bucket = CFG.get("s3_bucket", "dvc-bucket")

        existing = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
        if bucket not in existing:
            s3.create_bucket(Bucket=bucket)

    def upload_csv(self, csv_path) -> int:
        df = pd.read_csv(csv_path)
        df.to_sql(self.table, self.engine, if_exists="replace", index=False)
        return len(df)

    def load(self, query) -> pd.DataFrame:
        return pd.read_sql(query, self.engine)
