import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline

def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    g = df.get  # короче писать
    # Базовые новые признаки
    df["TotalSF"] = g("TotalBsmtSF").fillna(0) + g("1stFlrSF").fillna(0) + g("2ndFlrSF").fillna(0)
    df["TotalBath"] = (g("FullBath").fillna(0) + 0.5*g("HalfBath").fillna(0)
                       + g("BsmtFullBath").fillna(0) + 0.5*g("BsmtHalfBath").fillna(0))
    df["Age"] = g("YrSold") - g("YearBuilt")
    df["RemodAge"] = g("YrSold") - g("YearRemodAdd")
    df["IsRemodeled"] = (g("YearBuilt") != g("YearRemodAdd")).astype("float64")
    df["HasGarage"] = (g("GarageArea").fillna(0) > 0).astype("float64")
    df["HasBsmt"] = (g("TotalBsmtSF").fillna(0) > 0).astype("float64")
    df["HasFireplace"] = (g("Fireplaces").fillna(0) > 0).astype("float64")

    # Важно: MSSubClass трактуем как категорию
    if "MSSubClass" in df.columns:
        df["MSSubClass"] = df["MSSubClass"].astype(str)
    return df

def build_preprocessor() -> Pipeline:
    # 1) Добавляем фичи функцией, 2) Импутация+OHE/median по селекторам типов
    feat = ("feat", FunctionTransformer(_add_features, validate=False))
    num_tf = ("num",
              Pipeline([("impute", SimpleImputer(strategy="median"))]),
              selector(dtype_include=np.number))
    cat_tf = ("cat",
              Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
              selector(dtype_exclude=np.number))
    ct = ColumnTransformer([num_tf, cat_tf], remainder="drop")
    pre = Pipeline([feat, ("ct", ct)])
    return pre