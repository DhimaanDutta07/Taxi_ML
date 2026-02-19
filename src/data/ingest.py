import pandas as pd
from sqlalchemy import create_engine
from src.core.config import CFG


class DataLoader:
    def __init__(self, conn_url, table):
        self.conn_url = conn_url or CFG["postgres_url"]
        self.table = table
        self.engine = create_engine(self.conn_url)

    def upload_csv(self, csv_path) -> int:
        df = pd.read_csv(csv_path)
        df.to_sql(self.table, self.engine, if_exists="replace", index=False)
        return len(df)

    def load(self, query) -> pd.DataFrame:
        q = query
        return pd.read_sql(q, self.engine)
