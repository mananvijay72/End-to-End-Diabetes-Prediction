from src.logger import logging
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



class Accuracy:
    """
    Evaluation strategy that uses accuracy score
    """

    def __init__(self):
        pass
    
    def calculate_score(self, y_true, y_pred) -> float:
        """
        Args:
            y_true: 
            y_pred:
        Returns:
            accuracy: float
        """
        try:
            logging.info("Entered the calculate_score method of the MSE class")
            accuracy = accuracy_score(y_true, y_pred)
            logging.info("The Accuracy value is: " + str(accuracy))
            return accuracy
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the MSE class. Exception message:  "
                + str(e)
            )
            raise e


class Precision:
    """
    Evaluation strategy that uses precision
    """

    def __init__(self):
        pass

    def calculate_score(self, y_true, y_pred) -> float:
        """
        Args:
            y_true: 
            y_pred:
        Returns:
            precision: float
        """
        try:
            logging.info("Entered the calculate_score method of the R2Score class")
            precision = precision_score(y_true, y_pred)
            logging.info("The precision score value is: " + str(precision))
            return precision
        except Exception as e:
            
    
            logging.error(
                "Exception occurred in calculate_score method of the R2Score class. Exception message:  "
                + str(e)
            )
            raise e
        
class Recall:
    """
    Evaluation strategy that uses recall score
    """

    def __init__(self):
        pass

    def calculate_score(self, y_true, y_pred) -> float:
        """
        Args:
            y_true: 
            y_pred:
        Returns:
            recall: float
        """
        try:
            logging.info("Entered the calculate_score method of the R2Score class")
            recall = recall_score(y_true, y_pred)
            logging.info("The Recall value is: " + str(recall))
            return recall
        except Exception as e:
            
    
            logging.error(
                "Exception occurred in calculate_score method of the R2Score class. Exception message:  "
                + str(e)
            )
            raise e
        
class F1:
    """
    Evaluation strategy that uses F1 score
    """

    def __init__(self):
        pass

    def calculate_score(self, y_true, y_pred) -> float:
        """
        Args:
            y_true: 
            y_pred:
        Returns:
            f1: float
        """
        try:
            logging.info("Entered the calculate_score method of the R2Score class")
            f1 = f1_score(y_true, y_pred)
            logging.info("The F1 value is: " + str(f1))
            return f1
        except Exception as e:
            
            logging.error(
                "Exception occurred in calculate_score method of the F1Score class. Exception message:  "
                + str(e)
            )


