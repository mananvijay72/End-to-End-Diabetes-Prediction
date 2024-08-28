from src.pipeline.training_pipeline import train_pipeline
import mlflow

if __name__ == "__main__":
    train_pipeline(data_path=r"data\diabetes_prediction_dataset.csv")

