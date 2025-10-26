# House Prices — Advanced Regression (Kaggle)

End-to-end solution for the **House Prices: Advanced Regression** challenge.  
Reproducible sklearn pipeline with **log-target training**, solid **cross-validation**, and a simple **blend (LGBM + Ridge)**.

---

## Results
- **Public LB:** **0.12622** (RMSLE)
- **CV (log-RMSE):** _paste your last run output here, e.g._ `mean=0.13xx ± 0.00xx`

> We train on `log1p(SalePrice)` and evaluate RMSE in log space (≈ RMSLE). For submission we apply `expm1` back-transform.

---

## What worked
- Feature engineering: `TotalSF`, `TotalBath`, `Age`, `RemodAge`, `IsRemodeled`, `HasGarage/Bsmt/Fireplace`.
- Treat `MSSubClass` as **categorical** (cast to string).
- Robust preprocessing: `SimpleImputer` (median / most_frequent) + `OneHotEncoder(handle_unknown="ignore")`.
- **Blending**: LGBM (90%) + Ridge (10%) in **log space**, then `expm1`.

---

## Quickstart

### 0) Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn lightgbm xgboost joblib
