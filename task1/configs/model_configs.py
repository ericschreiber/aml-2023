from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor

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
    },
    'catboost':{
        # Catboost model.
        'model': CatBoostRegressor,
        'model_hyperparams': {
                    "iterations": 200,
                    "learning_rate": 1,
                    "depth": 2},
        'param_grid': {}
    },
    'xgboost':{
        # Xgboost model.
        'model': XGBRegressor,
        'model_hyperparams': {
                    "max_depth": 6,
                    "eta": 0.3},
        'param_grid': {}
    },
    'adaboost':{
        # Adaboost model.
        'model': AdaBoostRegressor,
        'model_hyperparams': {
                    "n_estimators": 500,
                    "learning_rate": 0.5},
        'param_grid': {}
    },

}