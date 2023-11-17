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
    f1_score, classification_report, recall_score, roc_auc_score,
    precision_recall_curve
)
# import calibration_curve
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibrationDisplay
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
experiment.log_code(file_name='07_test_eval_votingClassifier.py')

api = API(api_key=COMET_API_KEY)
#Download a Registry Model:
api.download_registry_model("2nd-milestone", "distance_model", "1.0.0",
                            output_path="./", expand=True)

api.download_registry_model("2nd-milestone", "angle_model", "1.0.0",
                            output_path="./", expand=True)

api.download_registry_model("2nd-milestone", "distance_angle_model", "1.0.0",
                            output_path="./", expand=True)

api.download_registry_model("2nd-milestone", "advanced_question2_model", "1.1.0",
                           output_path="./", expand=True)

api.download_registry_model("2nd-milestone", "best_model_xgrf_vote", "1.2.0",
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

    retain_season = data['season']
    data = data.drop(['season'],axis=1)
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
    
    data['season'] = retain_season

    # split the data
    train, val, test = utils.split_train_val_test(data)
    X_train = train.drop(columns=['season','is_goal'])
    y_train = train['is_goal']
    X_val = val.drop(columns=['season','is_goal'])
    y_val = val['is_goal']

    X_test = test.drop(columns=['season','is_goal'])
    y_test = test['is_goal']


    return X_train, y_train, X_val, y_val, X_test, y_test
#%%
def test_eval_7_1():
    #%%
    data_fe2 = pd.read_csv('data/new_data_for_modeling_tasks/df_data.csv') 

    #%%
    X_train, _, _, _, X_test, y_test = preprocess(data_fe2)
    
    #%%
    # Load pkl model
    logreg_dist = joblib.load('distance_model.pkl')
    logreg_angle = joblib.load('angle_model.pkl')
    logreg_dist_angle = joblib.load('distance_angle_model.pkl')
    xgb = joblib.load('advanced_question2_model.pkl')
    voting = joblib.load('best_model_xgrf_vote.pkl')
    
    #%%
    
    test = pd.concat([X_test, y_test], axis=1)
    test.dropna(axis=0,inplace=True)
    X_test = test.drop(columns=['is_goal'])
    y_test = test['is_goal']

    #%%
    model_names = ['Logistic_Distance_Model', 
                   'Logistic_Angle_Model', 
                   'Logistic_Distance_Angle_Model', 
                   'Best_XGBoost_Model',
                   'Best_Shot_Model']
    
    data_for_models = {'Logistic_Distance_Model': test[['shot_distance','is_goal']].copy(),
                       'Logistic_Angle_Model': test[['shot_angle','is_goal']].copy(),
                       'Logistic_Distance_Angle_Model': test[['shot_angle', 'shot_distance','is_goal']].copy(),
                       'Best_XGBoost_Model': test.copy(),
                       'Best_Shot_Model': test.copy()}
    
    models = [logreg_dist, logreg_angle, logreg_dist_angle, xgb, voting]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i, model in enumerate(models):
        print(model_names[i])
        data_for_pred = data_for_models[model_names[i]]
        # Predict probabilities
        y_pred = model.predict(data_for_pred.drop(columns=['is_goal']))
        data_for_pred['y_prob'] = model.predict_proba(data_for_pred.drop(columns=['is_goal']))[:, 1]

        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(data_for_pred['is_goal'], data_for_pred['y_prob'])
        roc_auc = roc_auc_score(data_for_pred['is_goal'], data_for_pred['y_prob'])

        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(data_for_pred['is_goal'], data_for_pred['y_prob'], n_bins=10)

        # Calculate goal rate and cumulative goals

        goal_rate = data_for_pred.groupby(pd.qcut(data_for_pred['y_prob'], 25))['is_goal'].mean() * 100

        # convert interval index to its midpoint values
        # goal_rate.index = goal_rate.index.map(lambda x: x.mid)
        goal_rate.index = goal_rate.index.map(lambda x: x.left)
        goal_rate.index = np.array(goal_rate.index) / goal_rate.index.max() * 100
        
        # get the cumulative sum of goals
        # cumulative_goals = val.groupby(pd.qcut(val['prob'], 10))[target].sum().cumsum()
        cumulative_goals = goal_rate[::-1].cumsum()/goal_rate[::-1].cumsum().max() * 100

        # Log metrics to Comet
        accuracy = accuracy_score(data_for_pred['is_goal'], y_pred)
        f1 = f1_score(data_for_pred['is_goal'], y_pred, average='macro')
        recall = recall_score(data_for_pred['is_goal'], y_pred, average='macro')

        line_style = '-'  # Replace with the line style you want to use
        marker = None  # Replace with the marker you want to use

        # Plot goal rate
        axs[0, 0].plot(goal_rate.index, goal_rate.values, label=f'{model_names[i]}', linestyle = line_style, marker = marker)
        axs[0,0].grid(True)

        axs[0, 1].plot(cumulative_goals.index, cumulative_goals.values, label=f'{model_names[i]}', linestyle = line_style, marker = marker)
        axs[0,1].grid(True)

        axs[1, 0].plot(fpr, tpr, label=f'{model_names[i]} (AUC = {roc_auc:.2f})', linestyle = line_style, marker = marker)
        axs[1,0].grid(True)

        CalibrationDisplay.from_predictions(data_for_pred['is_goal'], data_for_pred['y_prob'], n_bins=10, ax=axs[1, 1], label=f'{model_names[i]}')
        axs[1, 1].grid(True)
        print('model complete')
    # Finalize plots with appropriate labels, titles, and axis adjustments
    for ax in axs[0, :]:  # For top row subplots
        ax.invert_xaxis()
        ax.set_ylim([0, 100])

    for ax in axs[1, :]:  # For bottom row subplots
        ax.plot([0, 1], [0, 1], 'k--')

    # Set titles
    axs[0, 0].set_title('Goal Rate by Probability Percentile')
    axs[0, 1].set_title('Cumulative Goals by Probability Percentile')
    axs[1, 0].set_title('ROC Curve')
    axs[1, 1].set_title('Reliability Diagram')

    # set x and y labels
    axs[0, 0].set_xlabel('Model Probability Percentile')
    axs[0, 0].set_ylabel('Goal Rate (%)')

    axs[0, 1].set_xlabel('Model Probability Percentile')
    axs[0, 1].set_ylabel('Cumulative Goals (%)')

    axs[1, 0].set_xlabel('False Positive Rate')
    axs[1, 0].set_ylabel('True Positive Rate')

    axs[1, 1].set_xlabel('Model Probability')
    axs[1, 1].set_ylabel('Fraction of Positives')
    
    # Add legends
    for ax in axs.flatten():
        ax.legend(loc='best')

    # Adjust layout, save, and close the figure
    plt.tight_layout()
    plt.savefig('Test_Eval_5_Models.png')
    plt.close()

    # Log the figure to Comet
    experiment.log_image('Test_Eval_5_Models.png', name='Test Set Eval on 5 models')
    experiment.add_tags(["Test Set Evaluation", "Combined curves of 5 models"])

#%%
def test_eval_7_2():
    #%%
    test_logreg =  pd.read_csv('data/test_data/playoff_2020_Log_reg_Mile1_feat_name_test_data.csv')
    test_rest = pd.read_csv('data/test_data/playoff_2020_test_data_Task_5_6_Models.csv')
    data = pd.read_csv('data/new_data_for_modeling_tasks/df_data.csv')
    #%%
    # minmax scale test_logreg
    scaler = MinMaxScaler()
    test_logreg[['shot_distance','shot_angle']] = scaler.fit_transform(test_logreg[['shot_distance','shot_angle']])

    # preprocess test_rest
    _, _, _, _, X_test, y_test = preprocess(test_rest)
    X_train, _, _, _, _, _ = preprocess(data)

    #%%
    # check columns which are in X_train but not in X_test
    X_train_cols = X_train.columns
    X_test_cols = X_test.columns
    diff = X_train_cols.difference(X_test_cols)
    # all diff columns are categoricals and correspond to teams/shots which are not there in test set so adding them as 0s
    # add all diff columns to X_test and put value as 0
    for col in diff:
        X_test[col] = 0
    
    X_test = X_test[X_train_cols]
    #%%
    # Load pkl model
    logreg_dist = joblib.load('distance_model.pkl')
    logreg_angle = joblib.load('angle_model.pkl')
    logreg_dist_angle = joblib.load('distance_angle_model.pkl')
    xgb = joblib.load('advanced_question2_model.pkl')
    voting = joblib.load('best_model_xgrf_vote.pkl')

    #%%
    test = pd.concat([X_test, y_test], axis=1)
    test.dropna(axis=0,inplace=True)

    #%%
    model_names = ['Logistic_Distance_Model', 
                   'Logistic_Angle_Model', 
                   'Logistic_Distance_Angle_Model', 
                   'Best_XGBoost_Model',
                   'Best_Shot_Model']
    
    data_for_models = {'Logistic_Distance_Model': test_logreg[['shot_distance','is_goal']].copy(),
                       'Logistic_Angle_Model': test_logreg[['shot_angle','is_goal']].copy(),
                       'Logistic_Distance_Angle_Model': test_logreg[['shot_angle', 'shot_distance','is_goal']].copy(),
                       'Best_XGBoost_Model': test.copy(),
                       'Best_Shot_Model': test.copy()}
    
    models = [logreg_dist, logreg_angle, logreg_dist_angle, xgb, voting]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i, model in enumerate(models):
        print(model_names[i])
        data_for_pred = data_for_models[model_names[i]]
        # Predict probabilities
        y_pred = model.predict(data_for_pred.drop(columns=['is_goal']).values)
        data_for_pred['y_prob'] = model.predict_proba(data_for_pred.drop(columns=['is_goal']).values)[:, 1]

        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(data_for_pred['is_goal'], data_for_pred['y_prob'])
        roc_auc = roc_auc_score(data_for_pred['is_goal'], data_for_pred['y_prob'])

        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(data_for_pred['is_goal'], data_for_pred['y_prob'], n_bins=10)

        # Calculate goal rate and cumulative goals

        goal_rate = data_for_pred.groupby(pd.qcut(data_for_pred['y_prob'], 25))['is_goal'].mean() * 100

        # convert interval index to its midpoint values
        # goal_rate.index = goal_rate.index.map(lambda x: x.mid)
        goal_rate.index = goal_rate.index.map(lambda x: x.left)
        goal_rate.index = np.array(goal_rate.index) / goal_rate.index.max() * 100
        
        # get the cumulative sum of goals
        # cumulative_goals = val.groupby(pd.qcut(val['prob'], 10))[target].sum().cumsum()
        cumulative_goals = goal_rate[::-1].cumsum()/goal_rate[::-1].cumsum().max() * 100

        # Log metrics to Comet
        accuracy = accuracy_score(data_for_pred['is_goal'], y_pred)
        f1 = f1_score(data_for_pred['is_goal'], y_pred, average='macro')
        recall = recall_score(data_for_pred['is_goal'], y_pred, average='macro')

        line_style = '-'  # Replace with the line style you want to use
        marker = None  # Replace with the marker you want to use

        # Plot goal rate
        axs[0, 0].plot(goal_rate.index, goal_rate.values, label=f'{model_names[i]}', linestyle = line_style, marker = marker)
        axs[0,0].grid(True)

        axs[0, 1].plot(cumulative_goals.index, cumulative_goals.values, label=f'{model_names[i]}', linestyle = line_style, marker = marker)
        axs[0,1].grid(True)

        axs[1, 0].plot(fpr, tpr, label=f'{model_names[i]} (AUC = {roc_auc:.2f})', linestyle = line_style, marker = marker)
        axs[1,0].grid(True)

        CalibrationDisplay.from_predictions(data_for_pred['is_goal'], data_for_pred['y_prob'], n_bins=10, ax=axs[1, 1], label=f'{model_names[i]}')
        axs[1, 1].grid(True)
        print('model complete')
    # Finalize plots with appropriate labels, titles, and axis adjustments
    for ax in axs[0, :]:  # For top row subplots
        ax.invert_xaxis()
        ax.set_ylim([0, 100])

    for ax in axs[1, :]:  # For bottom row subplots
        ax.plot([0, 1], [0, 1], 'k--')

    # Set titles
    axs[0, 0].set_title('Goal Rate by Probability Percentile')
    axs[0, 1].set_title('Cumulative Goals by Probability Percentile')
    axs[1, 0].set_title('ROC Curve')
    axs[1, 1].set_title('Reliability Diagram')

    # set x and y labels
    axs[0, 0].set_xlabel('Model Probability Percentile')
    axs[0, 0].set_ylabel('Goal Rate (%)')

    axs[0, 1].set_xlabel('Model Probability Percentile')
    axs[0, 1].set_ylabel('Cumulative Goals (%)')

    axs[1, 0].set_xlabel('False Positive Rate')
    axs[1, 0].set_ylabel('True Positive Rate')

    axs[1, 1].set_xlabel('Model Probability')
    axs[1, 1].set_ylabel('Fraction of Positives')
    
    # Add legends
    for ax in axs.flatten():
        ax.legend(loc='best')

    # Adjust layout, save, and close the figure
    plt.tight_layout()
    plt.savefig('Test_Eval_5_Models_playoffs.png')
    plt.close()

    # Log the figure to Comet
    experiment.log_image('Test_Eval_5_Models_playoffs.png', name='Playoffs Test Set Eval on 5 models')
    experiment.add_tags(["Playoffs Test Set Evaluation", "Combined curves of 5 models"])

#%%
if __name__ == "__main__":
    test_eval_7_1()

    #%%
    experiment.end()