from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
    AdaBoostClassifier,
    RandomForestClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import RidgeClassifierCV, SGDClassifier, BayesianRidge
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process.kernels import RationalQuadratic

import numpy as np

RANDOM_SEED = 3141592

# lgbm = LGBMClassifier(n_estimators=2000, learning_rate=0.11, num_leaves=16, random_state=0, num_threads=128)
# xgboost = XGBClassifier(n_estimators=2000, random_state=0, learning_rate=0.11, max_depth=16, alpha=0.2)
# gradient = HistGradientBoostingClassifier(random_state=0, learning_rate=0.15, max_iter=400, max_leaf_nodes=31)
# forest = RandomForestClassifier(n_estimators=2000, random_state=0, n_jobs=-1)

model_configs = {
    "baseline": {
        # Default baseline model.
        "model": GradientBoostingClassifier,
        "model_hyperparams": {},
        "param_grid": {},
    },
    "histgradientboosting": {
        "model": HistGradientBoostingClassifier,
        "model_hyperparams": {
            "random_state": RANDOM_SEED,
            "max_iter": 400,
            "max_leaf_nodes": 32,
            "learning_rate": 0.15,
        },
        "param_grid": {},
    },
    "gbc": {
        "model": GradientBoostingClassifier,
        "model_hyperparams": {},
        "param_grid": {},
    },
    "catboost": {
        "model": CatBoostClassifier,
        "model_hyperparams": {},
        "param_grid": {},
    },
    "xgboost_optimised": {
        "model": XGBClassifier,
        "model_hyperparams": {
            "random_state": 0,
            "learning_rate": 0.1,
            "max_depth": 16,
            "colsample_bytree": 0.4,
            "gamma": 0.0,
            "min_child_weight": 1,
            "alpha": 0.5,
            "max_depth": 16,
            "n_estimators": 500,
        },
        "param_grid": {
            # "model__n_estimators": [400, 500, 600],# 1000, 2000, 3000],
            # "model__learning_rate": [0.05, 0.1, 0.15, 0.2],
            # "model__max_depth": [8, 16],
            # "model__alpha": [0.0, 0.01, 0.05, ]#0.1, 0.2, 0.3],
        },
    },
    "lightgbm_grid": {
        "model": LGBMClassifier,
        "model_hyperparams": {
            "random_state": RANDOM_SEED,
            "num_threads": 64,
            "objective": "multiclass",
        },
        "param_grid": {
            "model__n_estimators": [100, 200, 500, 1000],
            "model__learning_rate": [0.05, 0.1, 0.15, 0.2],
            "model__num_leaves": [
                8,
                16,
                32,
            ],
        },
    },
    "adaboost_grid3": {
        "model": AdaBoostClassifier,
        "model_hyperparams": {
            "n_estimators": 550,
            "learning_rate": 1,
            "estimator": DecisionTreeClassifier(max_depth=4),
        },
        "param_grid": {
            # "model__n_estimators": [400, 500, 800],
            # "model__learning_rate": [1, 1.5, 2],
        },
    },
    "svm": {
        "model": SVC,
        "model_hyperparams": {},
        "param_grid": {},
    },
    "gaussian_process_1": {
        "model": GaussianProcessClassifier,
        "model_hyperparams": {
            "kernel": RationalQuadratic(alpha=1, length_scale=1),
            "random_state": RANDOM_SEED,
            "n_restarts_optimizer": 5,
            "n_jobs": -1,
        },
        "param_grid": {},
    },
    "stacking": {
        "model": StackingClassifier,
        "model_hyperparams": {
            # "final_estimator": RidgeClassifierCV(),
            "cv": 5,
            "estimators": [
                ("gbc", GradientBoostingClassifier()),
                ("catboost", CatBoostClassifier()),
                ("xgboost", XGBClassifier()),
                ("lightgbm", LGBMClassifier()),
                ("adaboost", AdaBoostClassifier()),
                ("svm", SVC()),
                ("gaussian_process", GaussianProcessClassifier()),
                ("rf", RandomForestClassifier()),
                ("histgradientboosting", HistGradientBoostingClassifier()),
            ],
        },
        "param_grid": {},
    },
    "Viktor_stacking": {
        "model": StackingClassifier,
        "model_hyperparams": {
            "final_estimator": RidgeClassifierCV(),
            "cv": 5,
            "estimators": [
                (
                    "gbc",
                    GradientBoostingClassifier(
                        learning_rate=0.05,
                        n_estimators=500,
                        max_depth=7,
                        min_samples_split=60,
                        min_samples_leaf=9,
                        subsample=1.0,
                        max_features=50,
                        random_state=0,
                    ),
                ),
                (
                    "catboost",
                    CatBoostClassifier(depth=5, iterations=100, learning_rate=0.1),
                ),
                ("xgboost", XGBClassifier()),
                ("lightgbm", LGBMClassifier()),
                (
                    "adaboost",
                    AdaBoostClassifier(
                        n_estimators=550,
                        learning_rate=1,
                        estimator=DecisionTreeClassifier(max_depth=4),
                    ),
                ),
                ("svm", SVC()),
                ("gaussian_process", GaussianProcessClassifier()),
                ("rf", RandomForestClassifier(n_estimators=2000)),
                (
                    "histgradientboosting",
                    HistGradientBoostingClassifier(
                        max_iter=400, max_leaf_nodes=32, learning_rate=0.15
                    ),
                ),
            ],
        },
        "param_grid": {},
    },
    "randforest": {
        "model": RandomForestClassifier,
        "model_hyperparams": {
            "random_state": RANDOM_SEED,
            "n_jobs": 1,
            "n_estimators": 2000,
        },
        "param_grid": {},
    },
}
