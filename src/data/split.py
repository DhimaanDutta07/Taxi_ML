import pandas as pd
from sklearn.model_selection import train_test_split
from src.core.config import CFG

class DataSplitter():
    def split(self,df):
        df=df.copy()
        x=df.drop(CFG["target"],axis=1)
        y=df[CFG["target"]]
        return train_test_split(x,y,random_state=CFG["random_state"],test_size=CFG["test_size"])