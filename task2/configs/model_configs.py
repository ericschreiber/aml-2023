from sklearn.ensemble import GradientBoostingClassifier

import numpy as np

RANDOM_SEED = 3141592

model_configs = {
    "baseline": {
        # Default baseline model.
        "model": GradientBoostingClassifier,
        "model_hyperparams": {},
        "param_grid": {},
    },
}
