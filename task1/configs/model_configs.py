from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score

RANDOM_SEED = 3141592

model_configs = {
    "baseline": {
        # Default baseline model.
        "model": GradientBoostingRegressor,
        "model_hyperparams": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "subsample": 0.45,
        },
        "param_grid": {},
    },
    "catboost": {
        # Catboost model.
        "model": CatBoostRegressor,
        "model_hyperparams": {
            # "iterations": 200,
            # "learning_rate": 0.05,
            # "depth": 8,
            "eval_metric": "R2",
        },
        "param_grid": {
            "model__depth": [6, 8, 10],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__iterations": [30, 50, 100],
        },
    },
    "xgboost": {
        # Xgboost model.
        "model": XGBRegressor,
        "model_hyperparams": {},#"max_depth": 6, "eta": 0.3},
        "param_grid": {
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__min_child_weight": [ 1, 3, 5, 7],
            "model__learning_rate": [0.05, 0.10, 0.15],
            "model__gamma":[ 0.0, 0.1, 0.2],
            "model__colsample_bytree":[ 0.3, 0.4],
        },
    },
    "adaboost": {
        # Adaboost model.
        "model": AdaBoostRegressor,
        "model_hyperparams": {"n_estimators": 500, "learning_rate": 0.5},
        "param_grid": {},
    },
}
