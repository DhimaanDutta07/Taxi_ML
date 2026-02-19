from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.core.config import CFG
import pandas as pd
import numpy as np

class Preprocessor:
    def __init__(self, x):
        self.x = x
        self.num_cols = self.x.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = [c for c in self.x.columns if c not in self.num_cols]

    def build(self):
        num_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])

        pipe = ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.num_cols),
                ("cat", cat_pipe, self.cat_cols),
            ],
            remainder="drop"
        )
        return pipe

    def fit_save(self, pipe, x=None):
        if x is None:
            x = self.x

        X_t = pipe.fit_transform(x)

        if hasattr(X_t, "toarray"):
            X_t = X_t.toarray()

        cols = pipe.get_feature_names_out()
        X_df = pd.DataFrame(X_t, columns=cols)
        print(X_df)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.pre is None:
            raise RuntimeError("Preprocessor not built/fitted. Call fit() first.")
        Xt = self.pre.transform(X)
        cols = self.num_cols + self.cat_cols
        return pd.DataFrame(Xt, columns=cols)