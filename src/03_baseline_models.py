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
from sklearn.linear_model import LogisticRegression

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

#%%
# get api key from text file
COMET_API_KEY = open('comet_api_key.txt').read().strip()

# Create an experiment with your api key
experiment = Experiment(
    api_key=COMET_API_KEY,
    project_name="Baseline model",
    workspace="2nd milestone",
    log_code=True
)
experiment.log_code(file_name='03_baseline_models.py')
#%%
def baseline_question1(experiment):
    '''

    '''
    #%%
    data_baseline = pd.read_csv('data/baseline_model_data.csv')
    train_base, val_base, test_base = utils.split_train_val_test(data_baseline)

    #%%

    features = ['shot_distance']
    target = ['is_goal']
    #%%
    # Create the pipeline
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),  # This will scale features to the [0, 1] range
        ('logistic_regression', LogisticRegression())
    ])

    # Fit the pipeline
    pipeline.fit(train_base[features], train_base[target])

    #%%
    y_val = val_base[target]
    y_pred = pipeline.predict(val_base[features])

    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')
    recall = recall_score(y_val, y_pred, average='macro')
    experiment.log_metrics({"accuracy": accuracy, "f1_score": f1, "recall": recall})

    # Generate and log confusion matrix and classification report
    confusion_matrix_path = utils.plot_confusion_matrix(y_val, y_pred)
    classification_report_path = utils.plot_classification_report(y_val, y_pred)

    experiment.log_image(confusion_matrix_path, name='Confusion Matrix')
    experiment.log_image(classification_report_path, name='Classification Report')
    #%%
    model_reg_filename = f"03_baseline_models_question1.pkl"  # Modify as needed
    # Log the model to the experiment
    joblib.dump(pipeline, model_reg_filename)
    experiment.log_model(model_reg_filename, model_reg_filename)
    experiment.add_tags(["Baseline Logistic Question 1"])

#%%
def baseline_question2(experiment):
    #%%
    data_baseline = pd.read_csv('data/baseline_model_data.csv')
    train_base, val_base, test_base = utils.split_train_val_test(data_baseline)

    #%%

    features = ['shot_distance']
    target = ['is_goal']
    #%%
    # Create the pipeline
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),  # This will scale features to the [0, 1] range
        ('logistic_regression', LogisticRegression())
    ])

    # Fit the pipeline
    pipeline.fit(train_base[features], train_base[target])

    #%%
    model_reg_filename = f"03_baseline_models_question2.pkl"  # Modify as needed
    
    # Now call the function with this pipeline
    utils.plot_calibration_curve(model = pipeline, 
                                 features = features, 
                                 target = target, 
                                 val = val_base, 
                                 train = train_base, 
                                 model_reg_filename = model_reg_filename,
                                 tags = ["Baseline Logistic Question 2", "calibration_curve"], 
                                 experiment = experiment)

#%%
def baseline_question3(experiment):
    #%%
    class RandomModel:
        def __init__(self):
            pass
        
        def predict_proba(self, X):
            col_1 = np.random.uniform(size= len(X))
            col_2 = 1 - col_1
            return np.column_stack((col_1, col_2))
        
        def fit(self, X, y):
            pass
        def predict(self, X):
            return np.random.choice([0,1], size=len(X))
    
    #%%
    data_baseline = pd.read_csv('data/baseline_model_data.csv')
    train_base, val_base, test_base = utils.split_train_val_test(data_baseline)

    # Define features and model names
    features_list = [
        ['shot_distance'], 
        ['shot_angle'], 
        ['shot_angle', 'shot_distance'], 
        ['shot_distance']  # This will be used with the RandomModel
    ]
    target = ['is_goal']
    model_names = ['Distance Model', 'Angle Model', 'Combined Features Model', 'Random Model']

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i, features in enumerate(features_list):
        if model_names[i] == 'Random Model':
            model = RandomModel()
        else:
            model = Pipeline([
                ('scaler', MinMaxScaler()),
                ('logistic_regression', LogisticRegression())
            ])
            
        # Fit the model
        model.fit(train_base[features], train_base['is_goal'])

        # Predict probabilities
        y_true = val_base[target].values.ravel()
        val_base['prob'] = model.predict_proba(val_base[features])[:, 1]

        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_true, val_base['prob'])
        roc_auc = roc_auc_score(y_true, val_base['prob'])

        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(y_true, val_base['prob'], n_bins=10)

        # Calculate goal rate and cumulative goals
        goal_rate = val_base.groupby(pd.qcut(val_base['prob'], 25))[target].mean() * 100

        # convert interval index to its midpoint values
        # goal_rate.index = goal_rate.index.map(lambda x: x.mid)
        goal_rate.index = goal_rate.index.map(lambda x: x.left)
        goal_rate.index = np.array(goal_rate.index) / goal_rate.index.max() * 100
        
        # get the cumulative sum of goals
        # cumulative_goals = val.groupby(pd.qcut(val['prob'], 10))[target].sum().cumsum()
        cumulative_goals = goal_rate[::-1].cumsum()/goal_rate[::-1].cumsum().max() * 100

        # Log metrics to Comet
        accuracy = accuracy_score(y_true, model.predict(val_base[features]))
        f1 = f1_score(y_true, model.predict(val_base[features]), average='macro')
        recall = recall_score(y_true, model.predict(val_base[features]), average='macro')

        line_style = '-'  # Replace with the line style you want to use
        marker = None  # Replace with the marker you want to use

        # Plot goal rate
        axs[0, 0].plot(goal_rate.index, goal_rate.values, label=f'{model_names[i]}', linestyle = line_style, marker = marker)
        axs[0,0].grid(True)

        axs[0, 1].plot(cumulative_goals.index, cumulative_goals.values, label=f'{model_names[i]}', linestyle = line_style, marker = marker)
        axs[0,1].grid(True)

        axs[1, 0].plot(fpr, tpr, label=f'{model_names[i]} (AUC = {roc_auc:.2f})', linestyle = line_style, marker = marker)
        axs[1,0].grid(True)

        CalibrationDisplay.from_predictions(y_true, val_base['prob'], n_bins=10, ax=axs[1, 1], label=f'{model_names[i]}')
        axs[1, 1].grid(True)

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
    plt.savefig('all_metrics_combined.png')
    plt.close()

    # Log the figure to Comet
    experiment.log_image('all_metrics_combined.png', name='Combined Metrics Plot')
    experiment.add_tags(["Baseline Logistic Question 3", "Combined curves of 4 models"])

    #%%
if __name__ == "__main__":
    baseline_question1(experiment)
    baseline_question2(experiment)
    baseline_question3(experiment)

    #%%
    experiment.end()