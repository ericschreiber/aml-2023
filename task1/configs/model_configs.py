from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, ConstantKernel as C
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV

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
    "lgbmregressor_grid": {
        # LGBMRegressor model.
        "model": LGBMRegressor,
        "model_hyperparams": {},
        "param_grid": {
            "model__max_depth": [5, 6, 7, 8, 9],
            "model__num_leaves": [32, 64, 70, 90, 100, 128, 140], # between 2^(max_depth-1) and 2^max_depth
            "model__min_data_in_leaf": [10, 100, 500, 750, 1000, 1500, 2000], # to avoid overfitting
        },
    },
    "svr": {
        # SVR model.
        "model": SVR,
        "model_hyperparams": {},
        "param_grid": {},
    },
    "gpregressor": {
        # GaussianProcessRegressor model.
        "model": GaussianProcessRegressor,
        "model_hyperparams": {"kernel": RationalQuadratic()},
        "param_grid": {},
    },
    "stackingregressor": {
        # StackingRegressor model.
        "model": StackingRegressor,
        "model_hyperparams": {
            "final_estimator": RidgeCV(),
            "cv": 5,
            "estimators": [
                ("gbr", GradientBoostingRegressor()),
                ("svr", SVR(kernel="linear")),
                ("catboost", CatBoostRegressor()),
                ("xgboost", XGBRegressor()),
                ("adaboost", AdaBoostRegressor()),
                ("lgbmregressor", LGBMRegressor()),
                ("gpregressor", GaussianProcessRegressor(kernel=RationalQuadratic())),
            ],
        },
        "param_grid": {},
    },
    "stackingregressor_with_learned_parameter": {
        # StackingRegressor model.
        "model": StackingRegressor,
        "model_hyperparams": {
            "final_estimator": RidgeCV(),
            "cv": 5,
            "estimators": [
                ("gbr", GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, subsample=0.45)),
                ("svr", SVR(kernel=RationalQuadratic())),
                ("catboost", CatBoostRegressor(depth=6, learning_rate=0.1, iterations=100)),
                ("xgboost", XGBRegressor(learning_rate=0.1, max_depth=6, min_child_weight=1)),
                ("adaboost", AdaBoostRegressor()),
                ("lgbmregressor", LGBMRegressor(max_depth=6, num_leaves=45, min_data_in_leaf=10)),
                ("gpregressor", GaussianProcessRegressor(kernel=RationalQuadratic())),
            ],
        },
        "param_grid": {},
    },
}
