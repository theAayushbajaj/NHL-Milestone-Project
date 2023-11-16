#%%
# Basic libraries
import numpy as np
import pandas as pd
import warnings

# Preprocessing and model selection
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
import shap
from sklearn.base import clone
# Models
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

# Metrics
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc,
    f1_score, classification_report, recall_score,
    precision_recall_curve
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Hyperparameter optimization
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval

# Comet.ml for experiment tracking
from comet_ml import Experiment

# Utilities
import utils

# Joblib for model persistence
import joblib

#%%
# get api key from text file
COMET_API_KEY = open('comet_api_key.txt').read().strip()

# Create an experiment with your api key
experiment = Experiment(
    api_key=COMET_API_KEY,
    project_name="Advanced model",
    workspace="2nd milestone",
    log_code=True
)
experiment.log_code(file_name='05_advanced_models.py')


class CustomVotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, rf_best_params, xgb_best_params):
        self.random_forest = RandomForestClassifier(**rf_best_params)
        self.xgboost = XGBClassifier(**xgb_best_params)

        # Create a list of tuples with classifier name and classifier object
        self.classifiers = [
            ('random_forest', self.random_forest),
            ('xgboost', self.xgboost)
        ]

        # Initialize VotingClassifier with soft voting
        self.voting_classifier = VotingClassifier(estimators=self.classifiers, voting='soft')

    def fit(self, X_train, y_train):
        # Fit the voting classifier
        self.voting_classifier.fit(X_train, y_train)
        return self

    def predict(self, X):
        # Make predictions
        return self.voting_classifier.predict(X)

    def predict_proba(self, X):
        return self.voting_classifier.predict_proba(X)

#%%
def hyperparameter_tuning(model, X_train, y_train, X_val, y_val, space):
    def objective(params):
        model.set_params(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        loss = -f1_score(y_val, y_pred, average='macro')

        return {'loss': loss, 'status': STATUS_OK}
    
    # Initialize Trials object to keep track of results
    trials = Trials()

    # Run the optimization
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials
    )

    # After finding the best hyperparameters, log them
    best_hyperparams = space_eval(space, best)
    best_score = -trials.best_trial['result']['loss']

    # After finding the best hyperparameters, log them
    best_hyperparams = space_eval(space, best)
    best_score = -trials.best_trial['result']['loss']
    experiment.log_parameters(best_hyperparams)
    experiment.log_metric("best_score", best_score)

    return best_hyperparams
#%%
def best_shot():
    #%%
    # Load data
