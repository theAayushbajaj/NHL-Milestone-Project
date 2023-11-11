#%%

import numpy as np
import pandas as pd
import warnings
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from comet_ml import Experiment
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBClassifier


## import comet_ml at the top of your file
my_key = 'IGFjJ2mP1ZZPVHurdGupI5DJt'

# Create an experiment with your api key
experiment = Experiment(
    api_key=my_key,
    project_name="baseline model",
    workspace="2nd milestone",
)


#%%
# print current working directory
import os
print(os.getcwd())

#%%

# change working directory to parent
os.chdir('../')
print(os.getcwd())

#%%
import functions_mlstn_02 as _fct

#%%
# reset working directory
os.chdir('./05_Advanced_Models')
print(os.getcwd())


#%%

# load data
data = pd.read_csv('../data_for_remaining_tasks/df_data.csv')
data

#%%

# create data_sample, which is a subset of data, 100 rows
#data_sample = data.sample(n=100, random_state=1)
# to csv
#data_sample.to_csv('data_sample.csv', index=False)


# %%
# load the data used in baseline_models (part 3)
data_baseline = pd.read_csv('../03_baseline/baseline_model_data.csv')
# %%

train_base, val_base, test_base = _fct.split_train_val_test(data_baseline)

# %%

"""1. Train an XGBoost classifier using the same dataset using only the distance and angle features (similar to part 3). 
Don’t worry about hyperparameter tuning yet, this will just serve as a comparison to the baseline before we add more features. 
Add the corresponding curves to the four figures in your blog post. Briefly (few sentences) discuss your training/validation 
setup, and compare the results to the Logistic Regression baseline. Include a link to the relevant comet.ml entry for this experiment, 
but you do not need to log this model to the model registry.
"""

# import xgboost
import xgboost as xgb

features = ['shot distance', 'shot angle']
target = ['is goal']
# Define the pipeline
my_pipeline = Pipeline(steps=[
    ('scaler', MinMaxScaler()),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# Now call the function with this pipeline
_fct.plot_calibration_curve(my_pipeline, features, target, val_base, train_base, experiment)

# %%

"""2. Now, train an XGBoost classifier using all of the features you created in Part 4 and do some 
hyperparameter tuning to try to find the best performing model with all of these features. 
In your blog post, discuss your hyperparameter tuning setup, and include figures to substantiate 
your choice of hyperparameters. For example, you could select appropriate metrics and do a grid search 
with cross validation. Once tuned, include curves corresponding to the best model to the four figures 
in your blog post, and briefly compare the results to the XGBoost baseline. Include a link to the 
relevant comet.ml entry for this experiment, and log this model to the model registry.
"""
# get the type of each column
data.dtypes


#%%
# fix strength column
def fix_strength(df):
    strength = 'even'
    if df['num player home'] > df['num player away']:
        strength = 'power_play' if df['team shot'] == df['home team'] else 'short_handed'
    elif df['num player home'] < df['num player away']:
        strength = 'short_handed' if df['team shot'] == df['home team'] else 'power_play'
    df['strength'] = strength
    return df

#%%
data = data.apply(fix_strength, axis=1)
# change strength to categorical
#data['strength'] = data['strength'].astype('category')

#%%
data.dtypes



#%%

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from itertools import product
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix

# split the data

train, val, test = _fct.split_train_val_test(data)
X_train = train.drop(columns=['is goal'])
y_train = train['is goal']
X_val = val.drop(columns=['is goal'])
y_val = val['is goal']

# drop 'strength'
#X_train = X_train.drop(columns=['strength'])
#X_val = X_val.drop(columns=['strength'])

# Categorical columns and corresponding transformers
categorical_cols = X_train.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()

# remove 'game date' and 'period time'
categorical_cols.remove('game date')
categorical_cols.remove('period time')
# We need to convert booleans to integers before one-hot encoding
for col in categorical_cols:
    if X_train[col].dtype == 'bool':
        X_train[col] = X_train[col].astype(int)
        X_val[col] = X_val[col].astype(int)

#%%

# Define a custom transformer to parse the game date
def parse_game_date(X):
    X = pd.to_datetime(X['game date'], errors='coerce')
    return np.c_[X.dt.year, X.dt.month, X.dt.day]

# Define a custom transformer to parse the period time
def parse_period_time(X):
    period_time_in_seconds = X['period time'].str.split(':', expand=True).astype(int)
    period_time_in_seconds = period_time_in_seconds[0] * 60 + period_time_in_seconds[1]
    return period_time_in_seconds.values.reshape(-1, 1)

# Create transformers for the date and time columns
date_transformer = FunctionTransformer(parse_game_date, validate=False)
time_transformer = FunctionTransformer(parse_period_time, validate=False)


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Numerical columns and corresponding transformers
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])

time_pipeline = Pipeline(steps=[
    ('parse_time', time_transformer),
    ('scale', MinMaxScaler())
])

# Add the custom transformers to the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols),
        ('date', date_transformer, ['game date']),
        ('time', time_pipeline, ['period time']),
    ],
    remainder='drop'  # This drops the columns that we haven't transformed
)


# CLASS UNBALANCE WEIGHTS

# Compute class weight, this could be done manually as well
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(train['is goal']), 
    y=train['is goal']
)

# Calculate the scale_pos_weight for XGBoost, HOW MUCH MORE IMPORTANT IS THE POSITIVE CLASS
scale_pos_weight = class_weights[1] / class_weights[0]


#%%


for weight in [1.0, 2.0, 5.0, 10.0, 20.0]:
    # Initialize the Comet experiment
    experiment = Experiment(
        api_key=my_key,
        project_name='Advanced Models',
        workspace="2nd milestone",)

    # Include this in your XGBClassifier
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=weight))
    ])


    # Define your parameter grid as a dictionary
    # Note the 'model__' prefix to specify that these parameters belong to the 'model' step of the pipeline
    param_grid = {
        'model__max_depth': [3, 4, 5],
        'model__min_child_weight': [1, 5, 10],
        'model__learning_rate': [0.01, 0.1, 0.2],
        # Add more parameters as needed
    }

    # Create the possible combinations of parameters
    param_combinations = list(product(*param_grid.values()))

    # Store the best parameters and the best score
    best_params = None
    best_score = 0
    accuracy = 0

    # Iterate over all combinations
    for params in param_combinations:
        # Update the parameters of the pipeline
        params_dict = dict(zip(param_grid.keys(), params))
        model_pipeline.set_params(**params_dict)

        print(f"Evaluating {params_dict}...")
        
        # Fit the pipeline to the training data
        model_pipeline.fit(X_train, y_train)
        
        # Predict on the validation set using the pipeline
        y_val_pred = model_pipeline.predict(X_val)
        
        # Calculate the F1 macro average on the validation set
        current_score = f1_score(y_val, y_val_pred, average='macro')
        print(f"F1 Macro Average: {current_score}")
        accuracy_score = np.mean(y_val_pred == y_val)

        # EXPERIEMNT LOGGING
        # Log parameters, weight, and metrics for each model
        # experiment.log_parameters(params_dict)
        # experiment.log_parameter("scale_pos_weight", weight)
        # experiment.log_metric("f1_macro_average", current_score)
        # EXPERIEMNT LOGGING

        
        # If the current score is better than the best score, update the best score and best parameters
        if current_score > best_score:
            best_score = current_score
            best_params = params_dict
            accuracy = accuracy_score


    # Output the best score and the best parameters
    print(f"Best F1 Macro Average: {best_score}")
    print(f"Best parameters: {best_params}")
    print(f"Weight: {weight}")

    # EXPERIEMNT LOGGING
    # Log the best parameters and the best score
    experiment.log_parameters(best_params)
    experiment.log_parameter("scale_pos_weight", weight)
    #experiment.log_metric("best_f1_macro_average", best_score)
    #experiment.log_metric("accuracy", accuracy)
    experiment.log_metric("scale_pos_weight", weight)
    experiment.log_metric("hyperparamters", best_params)
    # all the metrics is the binary classification report


    # EXPERIEMNT LOGGING

    # Train the final pipeline on the full dataset with the best parameters
    model_pipeline.set_params(**best_params)
    model_pipeline.fit(X_train, y_train)
    # Classification report and confusion matrix
    y_pred = model_pipeline.predict(X_val)
    # Assuming you have the true labels 'y_val' and the predictions 'y_pred'
    experiment.log_other('classification_report', classification_report(y_val, y_pred))
    print(classification_report(y_val, y_pred))

    # Generate the classification report
    report_dict = classification_report(y_val, y_pred, output_dict=True)

    # Log each item from the classification report
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):  # Check if metrics is a dictionary
            for metric_name, value in metrics.items():
                # Construct a unique name for each metric including the label
                metric_label_name = f"{label}_{metric_name}"
                experiment.log_metric(metric_label_name, value)

    # confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    cm_filename = 'confusion_matrix.png'

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(cm_filename)  # Save it to a file
    plt.close()  # Close the plot to avoid displaying it inline if not needed

    # Log image to Comet.ml
    experiment.log_image(cm_filename)

    # plotting
    target = ['is goal']
    features = X_train.columns.tolist()
    _fct.plot_calibration_curve(model_pipeline, features, target, val, train, experiment)




    # If treating each weight as a separate experiment, end the experiment here
    experiment.end()
# EXPERIEMNT LOGGING




#%%

# expired code in cell 1 of `expired_code.ipynb`






