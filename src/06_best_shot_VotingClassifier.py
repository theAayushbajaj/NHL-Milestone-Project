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
    project_name="Best model",
    workspace="2nd milestone",
    log_code=True
)
experiment.log_code(file_name='05_advanced_models.py')

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
def preprocess(data):
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

    # split the data
    train, val, test = utils.split_train_val_test(data)
    X_train = train.drop(columns=['season','is_goal'])
    y_train = train['is_goal']
    X_val = val.drop(columns=['season','is_goal'])
    y_val = val['is_goal']

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

    return X_train, y_train, X_val, y_val
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
    data_fe2 = pd.read_csv('data/new_data_for_modeling_tasks/df_data.csv') 

    #%%
    X_train, y_train, X_val, y_val = preprocess(data_fe2)
    #%%
    # Perform hyperparameter tuning
    hp_space = {
        'rf_hp_space': {'n_estimators': hp.choice('n_estimators', range(100, 500)),
                        'max_depth': hp.choice('max_depth', range(5, 50)),
                        'min_samples_split': hp.uniform('min_samples_split', 0.01, 0.02),
                        'min_samples_leaf': hp.uniform('min_samples_leaf', 0.01, 0.02),
                        'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
                        'bootstrap': hp.choice('bootstrap', [True, False]),
                        'class_weight': hp.choice('class_weight', ['balanced', 'balanced_subsample']),
                        'criterion': hp.choice('criterion', ['gini', 'entropy'])},
        
        'xgb_hp_space': {'n_estimators': hp.choice('n_estimators', range(100, 500)),
                        'learning_rate': hp.quniform('learning_rate', 0.01, 0.2, 0.01),
                        'max_depth': hp.choice('max_depth', range(3, 14)),
                        'min_child_weight': hp.choice('min_child_weight', range(1, 10)),
                        'gamma': hp.uniform('gamma', 0.0, 0.5),
                        'subsample': hp.uniform('subsample', 0.5, 1.0),
                        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
                        'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
                        'reg_lambda': hp.uniform('reg_lambda', 1.0, 4.0),
                        'scale_pos_weight': hp.uniform('scale_pos_weight', 1.0, 10.0),
                        'max_delta_step': hp.choice('max_delta_step', range(1, 10))}
    }
    #X_train = X_train[~np.isnan(X_train)]
    #X_val = X_val[~np.isnan(X_val)]
    #model = RandomForestClassifier()
    #best_hyperparams_rf = hyperparameter_tuning(model,X_train, y_train, X_val, y_val, hp_space['rf_hp_space'])
    best_hyperparams_rf = {'bootstrap': True, 
                            'class_weight': None, 
                            'criterion' : 'gini',
                            'max_depth' :39,
                            'max_features' : 3, 
                            'min_samples_leaf' : 0.011565806463177194,
                            'min_samples_split' : 0.017239740311285764, 
                            'n_estimators' : 811}
    #model = XGBClassifier()
    #best_hyperparams_xgb = hyperparameter_tuning(model,X_train, y_train, X_val, y_val, hp_space['xgb_hp_space'])
    best_hyperparams_xgb = {'colsample_bytree': 0.882162152864482,
                            'gamma': 0.16487432807819533,
                            'learning_rate': 0.1,
                            'max_delta_step': 5,
                            'max_depth': 5,
                            'min_child_weight': 3,
                            'n_estimators': 334,
                            'reg_alpha': 0.617947326072274,
                            'reg_lambda': 3.271358327330144,
                            'scale_pos_weight': 4.001431354425059,
                            'subsample': 0.9752552528311702}
    
    #%%
    # Create the voting classifier
    train = pd.concat([X_train, y_train], axis=1)
    train.dropna(inplace=True)
    X_train = train.drop(columns=['is_goal'])
    y_train = train['is_goal']

    val = pd.concat([X_val, y_val], axis=1)
    val.dropna(inplace=True)
    X_val = val.drop(columns=['is_goal'])
    y_val = val['is_goal']

    model = CustomVotingClassifier(best_hyperparams_rf, best_hyperparams_xgb)
    model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    #%%
    # Plot calibration curve
    model_reg_filename = f"best_model_xgrf_vote.pkl"
    utils.plot_calibration_curve(model = model, 
                                 features = X_train.columns, 
                                 target = ['is_goal'], 
                                 val = pd.concat([X_val,y_val],axis=1), 
                                 train = X_train, 
                                 model_reg_filename = model_reg_filename,
                                 tags = ["Best Model","XGBoost RandomForest(default) voting", "calibration_curve"], 
                                 experiment = experiment,
                                 legend = 'RFXG Voting Classifier')
#%%
if __name__ == "__main__":
    best_shot()

    #%%
    experiment.end()