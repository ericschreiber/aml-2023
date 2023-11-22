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


import numpy as np

RANDOM_SEED = 3141592

model_configs = {
    "baseline": {
        # Default baseline model.
        "model": GradientBoostingClassifier,
        "model_hyperparams": {},
        "param_grid": {},
    },
    "histgradientboosting": {
        "model": HistGradientBoostingClassifier,
        "model_hyperparams": {},
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
    "xgboost": {
        "model": XGBClassifier,
        "model_hyperparams": {},
        "param_grid": {},
    },
    "lightgbm": {
        "model": LGBMClassifier,
        "model_hyperparams": {},
        "param_grid": {},
    },
    "adaboost": {
        "model": AdaBoostClassifier,
        "model_hyperparams": {},
        "param_grid": {},
    },
    "svm": {
        "model": SVC,
        "model_hyperparams": {},
        "param_grid": {},
    },
    "gaussian_process": {
        "model": GaussianProcessClassifier,
        "model_hyperparams": {},
        "param_grid": {},
    },
    "stacking": {
        "model": StackingClassifier,
        "model_hyperparams": {
            "final_estimator": RidgeClassifierCV(),
            "cv": 5,
            "estimators": [
                # ("gbc", GradientBoostingClassifier()),
                # ("catboost", CatBoostClassifier()),
                # ("xgboost", XGBClassifier()),
                # ("lightgbm", LGBMClassifier()),
                # ("adaboost", AdaBoostClassifier()),
                # ("svm", SVC()),
                # ("gaussian_process", GaussianProcessClassifier()),
                # ("rf", RandomForestClassifier()),
                # ("histgradientboosting", HistGradientBoostingClassifier()),
            ],
        },
        "param_grid": {},
    },
    "randforest": {
        "model": RandomForestClassifier,
        "model_hyperparams": {},
        "param_grid": {},
    },
}
