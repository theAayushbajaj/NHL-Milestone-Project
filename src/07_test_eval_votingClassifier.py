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

from comet_ml import API
#%%
# get api key from text file
COMET_API_KEY = open('comet_api_key.txt').read().strip()

# Create an experiment with your api key
experiment = Experiment(
    api_key=COMET_API_KEY,
    project_name="Test Set Eval",
    workspace="2nd milestone",
    log_code=True
)
experiment.log_code(file_name='05_advanced_models.py')

api = API(rest_api_key=COMET_API_KEY)
# Download a Registry Model:
api.download_registry_model("2nd-milestone", "best_model_xgrf_vote", "1.1.0",
                            output_path="./", expand=True)

#%%
class CustomVotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, rf_best_params, xgb_best_params):
        self.rf_best_params = rf_best_params
        self.xgb_best_params = xgb_best_params
        self.random_forest = RandomForestClassifier()
        self.xgboost = XGBClassifier(**self.xgb_best_params)

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
        # Set the classes_ attribute
        self.classes_ = self.voting_classifier.classes_
        return self

    def predict(self, X):
        # Make predictions
        return self.voting_classifier.predict(X)

    def predict_proba(self, X):
        return self.voting_classifier.predict_proba(X)
#%%
def preprocess_7_1(data):
    # omitting the non-essential features
    #data['attacking_goals'] = data.apply(lambda x: np.max(x['home goal'] - 1, 0) if x['home team'] == x['team shot'] else np.max(x['away goal']-1,0), axis = 1)
    #data['defending_goals'] = data.apply(lambda x: x['home goal'] if x['home team'] != x['team shot'] else x['away goal'], axis = 1)
    data['is_home'] = data.apply(lambda x: 1 if x['home_team_name'] == x['team_name'] else 0, axis = 1)

    data = data.drop(['game_date','game_id','Shooter','Goalie','rinkSide','home_goal','away_goal'],axis=1)
    def fix_strength(df):
        strength = 'even'
        if df['num_player_home'] > df['num_player_away']:
            strength = 'power_play' if df['team_name'] == df['home_team_name'] else 'short_handed'
        elif df['num_player_home'] < df['num_player_away']:
            strength = 'short_handed' if df['team_name'] == df['home_team_name'] else 'power_play'
        df['strength'] = strength
        return df

    data = data.apply(fix_strength, axis=1)
    
    def parse_period_time(row):
        minutes, seconds = row['periodTime'].split(':')
        period_time_in_seconds = int(minutes) * 60 + int(seconds)
        return period_time_in_seconds

    # Apply the function to the 'period time' column
    data['periodTime'] = data.apply(lambda x: parse_period_time(x), axis=1)

    # period as categorical
    data['period'] = data['period'].astype('category')
    # empty net as boolean
    data['emptyNet'] = data['emptyNet'].astype('bool')
    # is_home as boolean
    data['is_home'] = data['is_home'].astype('bool')

    # Convert booleans to integers
    bool_cols = data.select_dtypes(include=['bool']).columns
    data[bool_cols] = data[bool_cols].astype(int)

    # Handle categorical data
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns

    # Impute missing values with the most frequent value
    for col in categorical_cols:
        if data[col].isna().any():
            simple_imputer = SimpleImputer(strategy='most_frequent')
            data[col] = simple_imputer.fit_transform(data[[col]]).ravel()  # Use ravel() to convert the output to 1D array

    # One-hot encode categorical columns
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Handle numerical data
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns

    # Scale numerical columns
    scaler = MinMaxScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    # split the data
    train, val, test = utils.split_train_val_test(data)
    X_train = train.drop(columns=['season','is_goal'])
    y_train = train['is_goal']
    X_val = val.drop(columns=['season','is_goal'])
    y_val = val['is_goal']

    X_test = test.drop(columns=['season','is_goal'])
    y_test = test['is_goal']

    def imputer(data):
        # Convert booleans to integers
        bool_cols = data.select_dtypes(include=['bool']).columns
        data[bool_cols] = data[bool_cols].astype(int)

        # Handle categorical data
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns

        # Impute missing values with the most frequent value
        for col in categorical_cols:
            if data[col].isna().any():
                simple_imputer = SimpleImputer(strategy='most_frequent')
                data[col] = simple_imputer.fit_transform(data[[col]]).ravel()  # Use ravel() to convert the output to 1D array

        # One-hot encode categorical columns
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

        # Handle numerical data
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns

        # Scale numerical columns
        scaler = MinMaxScaler()
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

        return data
    
    X_train = imputer(X_train)
    X_val = imputer(X_val)
    X_test = imputer(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test
#%%
def test_eval_7_1():
    #%%
    data_fe2 = pd.read_csv('data/new_data_for_modeling_tasks/df_data.csv') 

    #%%
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_7_1(data_fe2)
    
    # Load pkl model
    model = joblib.load('best_model_xgrf_vote.pkl')
    #%%
    X_test = X_test[X_train.columns]
    utils.plot_calibration_curve(model = model, 
                                 features = X_train.columns, 
                                 target = ['is_goal'], 
                                 val = pd.concat([X_test,y_test],axis=1), 
                                 train = X_train,
                                 tags = ["Test Set Eval","Best Model","XGBoost RandomForest(default) voting", "calibration_curve"], 
                                 experiment = experiment,
                                 legend = 'RFXG Voting Classifier',
                                 model_reg_filename = None)
