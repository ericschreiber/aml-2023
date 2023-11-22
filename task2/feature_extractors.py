import numpy as np
import pandas as pd
import os


class BaseFeatureExtractor:
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X

    def transform_inputs(self, X_train, y_train, X_test):
        return X_train, y_train, X_test


class EricsBiobss:
    def __init__(self, n_estimators=100, n_jobs=-1):
        # Load pre-computed predictions in csv file task1/data/outliers_majority_vote_ocsvm_iforest_knn.csv
        # and store them in self.outlier_predictions
        basepath = os.path.realpath(__file__).split("feature_extractors.py")[0]
        self.X_train = pd.read_csv(
            os.path.join(basepath, "data/feature_extraction/bioss_X_train.csv"),
            index_col="id",
        )
        self.X_test = pd.read_csv(
            os.path.join(basepath, "data/feature_extraction/bioss_X_test.csv"),
            index_col="id",
        )
        self.y_train = pd.read_csv(
            os.path.join(basepath, "data/feature_extraction/bioss_y_train.csv"),
            index_col="id",
        )

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X

    def transform_inputs(self, X_train, y_train, X_test):
        # remove outliers from X and y (eg. the id's where self.outlier_predictions['outlier'] == 1.0)
        return self.X_train, self.y_train, self.X_test
