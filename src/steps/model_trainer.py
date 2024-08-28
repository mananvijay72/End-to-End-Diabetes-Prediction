import sys
from src.logger import logging

from typing import Tuple
from typing_extensions import Annotated

from src.steps.config import ModelConfig
from sklearn.ensemble import RandomForestClassifier
from src.components.model_trainer import RandomForest, HyperparameterTuner
import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin



def train_model(X_train:np.ndarray, y_train:pd.Series)-> ClassifierMixin:


    try:
        config = ModelConfig()

        if config.model =="RandomForestClassifier":
            model_class = RandomForest()
            base_model = model_class.base_model()

            if config.tuning == "True":
                
                try:
                    tuner = HyperparameterTuner()
                    parameters = config.parameter_grid

                    best_parameters = tuner.optimize(model=base_model, X_train = X_train, y_train = y_train, param_grid= parameters)
                    logging.info("best parameters found after cross validation")
                    trained_model = model_class.train(X_train=X_train, y_train= y_train, param_grid= best_parameters)
                    logging.info("Model Trainnig Done with best parameters")
                    return trained_model, best_parameters
                
                except Exception as e:
                    logging.error("Hypermater tuning failed")
                    raise e

            elif config.tuning == "False":

                trained_model = model_class.train(X_train=X_train, y_train= y_train)

                logging.info("Model Training done without hypermater tuning")
                return trained_model, {}

            else:
                logging.error("Wrong input in config")
        
        else:
            logging.error("Wrong model input in config")
        
    
    except Exception as e:
        logging.error("Train Model step failed")
        raise e


