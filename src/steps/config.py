
class ModelConfig:
    model = "RandomForestClassifier"
    tuning = "False"
    parameter_grid = {
    'n_estimators': [25, 50, 100, 150],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [3, 6, 9, 12],
    'max_leaf_nodes': [3, 6, 9, 15],
    'bootstrap': [True, False],
    'n_jobs' : [-1],
    }