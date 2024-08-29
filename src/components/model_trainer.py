from src.logger import logging
import sys
from typing import Tuple
from typing_extensions import Annotated

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

class RandomForest:

    def base_model(self):
        model = RandomForestClassifier()
        return model

    def train(self, X_train, y_train, param_grid = {}):
        '''
        Trains the data on a RandomForestModel
        Args:
            X_train. y_train: data to train model
            param_grid: parameters for hypermater tuning
        return:
            clf: a random forestr classifier
        '''

        try:
            clf = RandomForestClassifier(**param_grid)
            clf.fit(X_train, y_train)
            logging.info("Model trained")
            return clf
        
        except Exception as e:
            logging.error("Random Forest ClassifierModel trainig faled")
            raise e

class XGBoost:

    def base_model(self):
        model = XGBClassifier()
        return model

    def train(self, X_train, y_train, param_grid = {}):
        '''
        Trains the data on a XGBoost model
        Args:
            X_train. y_train: data to train model
            param_grid: parameters for hypermater tuning
        return:
            clf: a XGBosstor classifier
        '''

        try:
            clf = XGBClassifier(**param_grid)
            clf.fit(X_train, y_train)
            logging.info("Model trained")
            return clf
        
        except Exception as e:
            logging.error("XGboost classifier trainig faled")
            raise e

class HyperparameterTuner:

    def __init__(self):
        pass

    def optimize(self, model, X_train, y_train, param_grid:dict) -> dict:

        try:
            optimizer = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs= -1, verbose=1)
            optimizer.fit(X_train, y_train)
            best_parameters = optimizer.best_params_
            logging.info(f"Hypermater tuning done for {model} with best parameters as {best_parameters}")
            return best_parameters

        except Exception as e:
            logging.error(f"Failed Hypermater Tuning {type(model)}")
            raise e