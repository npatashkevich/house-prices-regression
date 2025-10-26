import os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from features import build_preprocessor

DATA_PATH = "data/train.csv"
ART_DIR = "artifacts"
os.makedirs(ART_DIR, exist_ok=True)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def main():
    df = pd.read_csv(DATA_PATH)

    # лог-таргет
    y = np.log1p(df["SalePrice"].values)
    X = df.drop(columns=["SalePrice", "Id"], errors="ignore")

    # ── CV на LGBM (для отчёта)
    pipe_cv = Pipeline([
        ("pre", build_preprocessor()),
        ("model", LGBMRegressor(
            random_state=42,
            n_estimators=900,
            learning_rate=0.03,
            num_leaves=63,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            min_child_samples=30,
            reg_alpha=0.1,
            reg_lambda=0.5,
            verbosity=-1,
        )),
    ])
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(rmse, greater_is_better=False)  # neg RMSE на лог-таргете ≈ RMSLE
    scores = -cross_val_score(pipe_cv, X, y, cv=cv, scoring=scorer, n_jobs=-1)
    print(f"CV RMSE_log: mean={scores.mean():.4f} ± {scores.std():.4f}")

    # ── Финальная подгонка двух моделей на всём train
    pipe_lgbm = Pipeline([
        ("pre", build_preprocessor()),
        ("model", LGBMRegressor(
            random_state=42,
            n_estimators=900,
            learning_rate=0.03,
            num_leaves=63,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            min_child_samples=30,
            reg_alpha=0.1,
            reg_lambda=0.5,
            verbosity=-1,
        )),
    ]).fit(X, y)

    pipe_ridge = Pipeline([
        ("pre", build_preprocessor()),
        ("model", Ridge(alpha=3.0)),
    ]).fit(X, y)

    # ── Сохраняем оба пайплайна
    joblib.dump(pipe_lgbm, os.path.join(ART_DIR, "model_lgbm.joblib"))
    joblib.dump(pipe_ridge, os.path.join(ART_DIR, "model_ridge.joblib"))
    print("Saved:", os.path.join(ART_DIR, "model_lgbm.joblib"))
    print("Saved:", os.path.join(ART_DIR, "model_ridge.joblib"))

if __name__ == "__main__":
    main()