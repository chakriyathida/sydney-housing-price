"""Re-train the deployed model from the collected dataset and save it to app/model.joblib.

The notebook (Part 3) already saves this file. Run this script only if the saved model
cannot be loaded on your machine (e.g. a different scikit-learn version):

    python app/train_model.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from housing_features import HousingFeatures, load_raw  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"


def train(data_path=ROOT / "Sydney_Housing_Data.xlsx", save_path=MODEL_PATH):
    X, price, _ = load_raw(data_path)
    y = np.log(price)
    pipe = Pipeline([("features", HousingFeatures(interactions=True)),
                     ("model", RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1))])
    grid = GridSearchCV(pipe, {"model__min_samples_leaf": [1, 3], "model__max_features": [0.33, 0.6, 1.0]},
                        cv=KFold(5, shuffle=True, random_state=1), scoring="neg_root_mean_squared_error")
    grid.fit(X, y)

    # 80% prediction range from out-of-fold residuals, separately for houses and strata
    oof = cross_val_predict(grid.best_estimator_, X, y, cv=KFold(5, shuffle=True, random_state=42))
    resid = pd.Series(y.values - oof)
    seg = np.where(X["Type"].isin(["House", "Semi/Duplex"]), "House", "Strata")
    interval = {s: (float(resid[seg == s].quantile(0.10)), float(resid[seg == s].quantile(0.90)))
                for s in ["House", "Strata"]}
    rmse = float(np.sqrt(np.mean((y.values - oof) ** 2)))
    bundle = {"pipeline": grid.best_estimator_, "interval": interval, "model_name": "Random forest",
              "cv_summary": {"RMSE (log)": rmse}, "sklearn_version": sklearn.__version__}
    joblib.dump(bundle, save_path)
    return bundle


if __name__ == "__main__":
    b = train()
    print("Saved", MODEL_PATH, "| best params:", b["pipeline"].named_steps["model"].get_params()["max_features"],
          "| CV RMSE (log):", round(b["cv_summary"]["RMSE (log)"], 3))
