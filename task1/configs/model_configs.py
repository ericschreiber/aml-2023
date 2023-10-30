from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

RANDOM_SEED = 3141592

model_configs = {
    'baseline':{
        # Default baseline model.
        'model': GradientBoostingRegressor,
        'model_hyperparams': {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "subsample": 0.45},
        'param_grid': {}
    }
}