# House Prices — Advanced Regression (Kaggle)

End-to-end solution for the **House Prices: Advanced Regression** challenge.  
Reproducible sklearn pipeline with **log-target training**, solid **cross-validation**, and a simple **blend (LGBM + Ridge)**.

---

## Results
- **Public LB:** **0.12622** (RMSLE)
- **CV (log-RMSE):** mean=0.1351 ± 0.0188

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
~~~bash
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn lightgbm xgboost joblib
~~~

### 1) Data
Place Kaggle files into `./data/`:
~~~
data/
  ├─ train.csv
  └─ test.csv
~~~

### 2) Train
~~~bash
python src/train.py
# prints: "CV RMSE_log: mean=... ± ..."
# saves:  artifacts/model_lgbm.joblib, artifacts/model_ridge.joblib
~~~

### 3) Predict (build submission)
~~~bash
python src/predict.py
# saves: artifacts/submissions/submission_YYYYMMDD_HHMM.csv
~~~

Upload the CSV on Kaggle → **Submit Predictions**.

---

## Repo structure
~~~
src/
  features.py      # preprocessing + FE (TotalSF, TotalBath, etc.)
  train.py         # log-target CV, fit LGBM & Ridge, save both pipelines
  predict.py       # blend in log space (0.9/0.1), expm1, build submission
artifacts/
  model_lgbm.joblib
  model_ridge.joblib
  submissions/
data/               # train.csv, test.csv (git-ignored)
README.md
.gitignore
~~~

## Reproducibility & Safety
- Single `Pipeline(preprocess → model)` to avoid leakage.
- `predict.py` guards against NaN/inf and non-positive prices.
- Seeds fixed where applicable.

## Next steps (optional)
- Tune blend weights by OOF CV; add XGB to the blend.
- Auto log-transform for highly skewed numeric features.
- Permutation importances / SHAP in a short report notebook
