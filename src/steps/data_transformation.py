from src.logger import logging
import sys
from typing import Tuple
from typing_extensions import Annotated

import pandas as pd
import numpy as np
from src.components.data_transformation import DataCleaning, PreProcessor
from sklearn.model_selection import train_test_split



def transform(data: pd.DataFrame) -> Tuple[
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]]:
    
    '''
    Cleans, preprocess balances and splits the data for training
    
    Args:
        df: Ingested data from source(pandas dataframe)
    
    Returns:
        X_train, X_test, y_train, y_test
    '''
    try:
        clean_data = DataCleaning().clean_data(df=data)
        preprocessor = PreProcessor()

        features, target = preprocessor.preprocess(clean_data)

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

        logging.info("Data transformation completed")

        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logging.error("Failed to Transform data")
        raise e



