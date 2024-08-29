# Diabetes Prediction Project

## Project Overview
This project focuses on predicting diabetes using machine learning techniques. The project involved data cleaning, balancing the dataset, performing Exploratory Data Analysis (EDA), and applying predictive analytics using seven different models. After model selection, hyperparameter tuning was conducted to achieve an F1 score of 0.97. The entire process was tracked using MLflow, and the final model was deployed on Streamlit Cloud using a CI/CD pipeline with GitHub Actions.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling and Evaluation](#modeling-and-evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Tracking with MLflow](#model-tracking-with-mlflow)
- [Deployment with Streamlit](#deployment-with-streamlit)
- [CI/CD Pipeline](#cicd-pipeline)
- [How to Run the Project](#how-to-run-the-project)
- [Conclusion](#conclusion)

## Data Cleaning and Preprocessing
- Handled missing values and outliers.
- Scaled features using standardization.
- Balanced the dataset using [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) to address class imbalance.

## Exploratory Data Analysis (EDA)
- Conducted EDA to understand feature distributions and relationships.
- Visualized data using libraries such as `matplotlib` and `seaborn`.
- Identified key features influencing diabetes prediction.

## Modeling and Evaluation
- Applied seven different machine learning models, including:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - k-Nearest Neighbors (k-NN)
  - Gradient Boosting
  - XGBoost
  - Neural Networks
- Evaluated models using metrics such as Accuracy, Precision, Recall, and F1 Score.



Project Title: Diabetes Prediction with Streamlit Deployment

Description:

This project tackles the challenge of predicting diabetes using machine learning. We built a robust model pipeline that incorporates data cleaning, balancing, exploratory data analysis (EDA), and predictive analysis using various models.

Key Features:

Data Preprocessing: Thorough data cleaning, including handling missing values and outliers, for reliable modeling.
Data Balancing: Effective measures to address class imbalances and ensure model fairness.
Exploratory Data Analysis (EDA): Deep understanding of the data through visualizations and statistical analysis to uncover patterns and relationships relevant to diabetes prediction.
Predictive Modeling: Implementation of seven different machine learning algorithms to compare and select the most suitable for diabetes prediction.
Hyperparameter Tuning: Optimization of model hyperparameters for improved performance, achieving an F1 score of 0.97.
Experiment Tracking with MLflow: Efficient model experimentation management, enabling easy comparison and reproducibility.
Streamlit Web App: User-friendly interface for interactive testing and demonstration of the model.
CI/CD Pipeline with GitHub Actions: Automated deployment of the model on Streamlit Cloud, streamlining the transition from development to production.
Technologies Used:

Python
Pandas
NumPy
Scikit-learn (or other chosen ML libraries)
Matplotlib/Seaborn (for visualization)
MLflow (for experiment tracking)
Streamlit (for web app creation)
GitHub Actions (for CI/CD pipeline)
Project Structure:

project/
├── data/
│   ├── raw/
│   │   └── diabetes_data.csv
│   └── processed/
│       └── diabetes_processed.csv
├── notebooks/
│   ├── data_cleaning.ipynb
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── hyperparameter_tuning.ipynb
├── models/
│   └── best_model.pkl
├── mlflow/
│   └── mlruns  # Stores MLflow experiment data
├── app.py  # Streamlit web app code
├── requirements.txt  # Dependency list
└── README.md  # Project documentation (this file)
Running the Project:

Clone the Repository:

Bash
git clone https://github.com/<your-username>/diabetes-prediction.git
Use code with caution.

Install Dependencies:

Bash
cd diabetes-prediction
pip install -r requirements.txt
Use code with caution.

Explore Notebooks (Optional):

Open the Jupyter notebooks in this directory to replicate the data cleaning, EDA, model training, and hyperparameter tuning stages.

Run Streamlit Web App:

Bash
streamlit run app.py
Use code with caution.

This will launch the Streamlit web app, allowing you to interact with the model.

Deployment (Streamlit Cloud):

This section provides detailed instructions on deploying the project to your Streamlit Cloud account (replace placeholders with your actual information).

Set up a Streamlit Cloud account (if you haven't already).
Create a new app in Streamlit Cloud.
Push the code to your GitHub repository.
Configure your GitHub Actions workflow by creating a .github/workflows directory and adding a YAML file that specifies the deployment steps. Reference the Streamlit Cloud deployment action (streamlit/actions-deploy@v1) and provide your streamlit_cloud_url and a GitHub secret named STREAMLIT_CLOUD_ACCESS_TOKEN (generated from your Streamlit Cloud account settings).
On push to your main branch, GitHub Actions will automatically trigger the CI/CD pipeline, deploying your Streamlit app to Streamlit Cloud.
