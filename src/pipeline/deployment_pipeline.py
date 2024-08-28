import numpy as np
import pandas as pd
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer

from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from src.steps.data_ingestion import ingest_data
from src.steps.data_transformation import transform
from src.steps.model_trainer import train_model
from src.steps.model_evauation import evaluation

# Define the pipeline

docker_settings = DockerSettings(required_integrations=[MLFLOW])

@pipeline(enable_cache=True, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy:float = 0.9,
    workers:int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    # Define the steps
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = transform(df)
    trained_model = train_model(X_train, y_train)
    accuracy, precision, recall, f1 = evaluation(trained_model, X_test, y_test)

    deployment_decision = (float(accuracy) >= min_accuracy)


    # Deploy the model to mlflow
    mlflow_model_deployer_step(
        model_name="diabetes_model",
        model = trained_model,
        deployment_decision = deployment_decision,
        workers = workers,
        timeout = timeout,
    )