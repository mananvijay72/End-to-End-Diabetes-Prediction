from src.logger import logging
import sys
from typing import Tuple
from typing_extensions import Annotated
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE



class DataCleaning:
    '''
    Clean the ingested data
    '''
    def __init__(self):
        pass

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Cleans the data.
        Args:
            df: data frame to clean
        Returns:
            Cleaned data frame
        '''

        try: 
            #replacing correct values in smoking_history column
            df[['smoking_history']] = df[['smoking_history']].replace({'never':'non_smoker',
                                                       'No Info': 'non_smoker',
                                                       'former': 'past_smoker',
                                                       'current': 'current_smoker',
                                                       'not current': 'past_smoker',
                                                       'ever' : 'past_smoker'})
            #removing duplicate values
            df = df.drop_duplicates()
            #removing unknown values
            df = df[df['gender'] != 'Other']

            logging.info("Data is cleaned")

            return df
        
        except Exception as e:
            logging.error("failed to clean data")
            raise e

class PreProcessor:
    '''
    Preprocess the data - Scaling, Onehot Encoding,Balancing(Upsampling SMOTE)
    '''
    def __init__(self):
        pass

    def preprocess(self, df:pd.DataFrame) -> Tuple[Annotated[np.ndarray, "X"],
                                                   Annotated[pd.Series, "y"]]:
        
        '''
        Args:
            df: cleaned data
        Returns:    
            X(features),y(target)
            Scaled, one hot encoded and balanced data
        '''
        try:
            X = df.drop("diabetes", axis=1)
            y = df["diabetes"]
            
            #preprocess pipeling for scaling and oneHotEncoding the selected columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level','hypertension','heart_disease']),
                    ('cat', OneHotEncoder(), ['gender','smoking_history'])
                ])
            
            balancing = SMOTE(random_state=42)

            X = preprocessor.fit_transform(X)

            X, y = balancing.fit_resample(X,y)

            #saving preprocessor
            with open("artifacts/preprocessor.pkl", "wb") as handle:
                pickle.dump(preprocessor, handle)

            logging.info("Preprocessing of the data done.")

            return (X, y)
        
        except Exception as e:
            logging.error("Failed to preprocess the data")
            raise e
    

