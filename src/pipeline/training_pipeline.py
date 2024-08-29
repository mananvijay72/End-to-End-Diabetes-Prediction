from src.steps.data_ingestion import ingest_data
from src.steps.data_transformation import transform
from src.steps.model_trainer import train_model
from src.steps.model_evauation import evaluation
from src.steps.config import ModelConfig
import pickle
from src.logger import logging
import mlflow


def train_pipeline(data_path: str):

    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = transform(df)
    logging.info("Data transformation completed")
    logging.info("Training model")
    trained_model = train_model(X_train, y_train)
    logging.info("Model training completed")
    logging.info("Evaluating model")
    accuracy, precision, recall, f1 = evaluation(trained_model, X_test, y_test)
    print("MODEL: ", trained_model.__class__.__name__)
    print("ACCURACY: ", accuracy)
    print("PRECISION: ", precision)
    print("RECALL: ", recall)
    print("F1: ", f1)
    logging.info("Training pipeline completed")


    mlflow.set_experiment("Diabetese Prediction 1")
    with mlflow.start_run(run_name=trained_model.__class__.__name__):

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(trained_model, "model")


    logging.info("Model logged to mlflow")

    #savibg the model in artifacts
    with open('artifacts/model.pkl', 'wb') as handle:
        pickle.dump(trained_model, handle)
    return trained_model