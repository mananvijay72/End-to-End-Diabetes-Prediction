from src.logger import logging
import sys

import pandas as pd

class IngestData:
    '''
    Class to ingest the data
    Args:
        data_path: path of the data to ingested
    '''


    def get_data(self, data_path:str) -> pd.DataFrame:

        df = pd.read_csv(data_path)
        return df
    

def ingest_data(data_path: str) -> pd.DataFrame:
    '''
    Args:
        data_path: path of the data to ingest
    Returns:
        data as a pandas dataframe
    '''
    try:
        ingest_data = IngestData()
        data = ingest_data.get_data(data_path)
        logging.info(f"Data ingested from: {data_path}")
        return data
    
    except Exception as e:
        logging.error(f"Failed to ingest data from: {data_path}")
        raise e
    

