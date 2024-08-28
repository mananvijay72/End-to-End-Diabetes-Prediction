from src.logger import logging
from typing import Tuple
from typing_extensions import Annotated
from sklearn.base import ClassifierMixin
from src.components.model_evaluaion import Accuracy, Precision, Recall, F1
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score



def evaluation(model: ClassifierMixin, X_test: np.ndarray, y_test:pd.Series
        ) -> Tuple[Annotated[float, "accuracy"], Annotated[float, "precision"], Annotated[float, "recall"], Annotated[float, "f1"]]:

    """
    Args:
        x_test: pd.DataFrame
        y_test: pd.Series
    Returns:
        r2_score: float
        rmse: float
    """
    try:
        
        logging.info("entered evaluation step")
        prediction = model.predict(X_test)

        # Using the MSE class for mean squared error calculation
        accuracy_class = Accuracy()
        accuracy = accuracy_class.calculate_score(y_test, prediction)

        # Using the Precision to calculate precsion score
        precision_class = Precision()
        precision = precision_class.calculate_score(y_test, prediction)

        # Using the Recall class for recall calculation
        recall_class = Recall()
        recall = recall_class.calculate_score(y_test, prediction)

        f1_class = F1()
        f1 = f1_class.calculate_score(y_test, prediction)

        logging.info("Evaluation step completed")
        return accuracy, precision, recall, f1
     

    except Exception as e:
        logging.error("Failed the evaluation step")
        raise e
