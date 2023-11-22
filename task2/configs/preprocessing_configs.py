from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, f_classif

import numpy as np

from feature_extractors import BaseFeatureExtractor, EricsBiobss

preprocessing_configs = {
    "baseline": {
        # Default baseline preprocessing
        "order": ["feature_extractor", "imputer", "scaler", "selector"],
        "feature_extractor": BaseFeatureExtractor,
        "feature_extractor_hyperparams": {},
        "imputer": SimpleImputer,
        "imputer_hyperparams": {"strategy": "median"},
        "scaler": StandardScaler,
        "scaler_hyperparams": {},
        "selector": SelectKBest,
        "selector_hyperparams": {
            "score_func": f_classif,
            "k": 100,
        },
        "param_grid": {},
    },
    "ericBioss": {
        "order": ["feature_extractor", "imputer", "scaler", "selector"],
        "feature_extractor": EricsBiobss,
        "feature_extractor_hyperparams": {},
        "imputer": SimpleImputer,
        "imputer_hyperparams": {
            "strategy": "median",
            "missing_values": np.nan,
        },
        "scaler": StandardScaler,
        "scaler_hyperparams": {},
        "selector": SelectKBest,
        "selector_hyperparams": {
            "score_func": f_classif,
            "k": 75,
        },
        "param_grid": {},
    },
}
