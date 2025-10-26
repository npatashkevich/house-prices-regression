import os, joblib, numpy as np, pandas as pd

DATA_TEST = "data/test.csv"
ART_DIR = "artifacts"
SUB_DIR = os.path.join(ART_DIR, "submissions")
os.makedirs(SUB_DIR, exist_ok=True)

def main():
    # загружаем обе модели
    m1 = joblib.load(os.path.join(ART_DIR, "model_lgbm.joblib"))
    m2 = joblib.load(os.path.join(ART_DIR, "model_ridge.joblib"))

    test = pd.read_csv(DATA_TEST)
    X_test = test.drop(columns=["Id"], errors="ignore")

    # блендинг лог-предсказаний
    y_log = 0.9 * m1.predict(X_test) + 0.1 * m2.predict(X_test)
    y = np.expm1(y_log)

    # safety: no inf/NaN/<=0
    y = np.where(np.isfinite(y), y, np.nan)
    y = np.nan_to_num(y, nan=np.nanmedian(y))
    y = np.clip(y, 1.0, None)

    sub = pd.DataFrame({"Id": test["Id"], "SalePrice": y})
    out = os.path.join(SUB_DIR, f"submission_{pd.Timestamp.now():%Y%m%d_%H%M}.csv")
    sub.to_csv(out, index=False)
    print("Saved:", out)

if __name__ == "__main__":
    main()