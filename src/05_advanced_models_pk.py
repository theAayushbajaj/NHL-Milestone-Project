#%%

import numpy as np
import pandas as pd
import warnings
from sklearn.calibration import calibration_curve
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, \
            classification_report, recall_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from comet_ml import Experiment
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from itertools import product
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
from xgboost import XGBClassifier
import utils
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval

#%%
COMET_API_KEY = open('comet_api_key.txt').read().strip()

# Create an experiment with your api key
experiment = Experiment(
    api_key=COMET_API_KEY,
    project_name="Advanced model",
    workspace="2nd milestone",
)


#%%

# load data
data = pd.read_csv('data/data_for_remaining_tasks/df_data.csv')
data.head()



# %%
# load the data used in baseline_models (part 3)
data_baseline = pd.read_csv('data/baseline_model_data.csv')
# %%

train_base, val_base, test_base = utils.split_train_val_test(data_baseline)

# %%

"""1. Train an XGBoost classifier using the same dataset using only the distance and angle features (similar to part 3). 
Don’t worry about hyperparameter tuning yet, this will just serve as a comparison to the baseline before we add more features. 
Add the corresponding curves to the four figures in your blog post. Briefly (few sentences) discuss your training/validation 
setup, and compare the results to the Logistic Regression baseline. Include a link to the relevant comet.ml entry for this experiment, 
but you do not need to log this model to the model registry.
"""

# import xgboost
import xgboost as xgb

features = ['shot_distance', 'shot_angle']
target = ['is_goal']
# Define the pipeline
my_pipeline = Pipeline(steps=[
    ('scaler', MinMaxScaler()),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# Now call the function with this pipeline
utils.plot_calibration_curve(my_pipeline, features, target, val_base, train_base, experiment)

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

# split the data

train, val, test = utils.split_train_val_test(data)
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

# Create the preprocessing and modeling pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])



#%%
def objective(params):
    # Create the RandomForestClassifier with the given hyperparameters
    model_pipeline.set_params(**params)
    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_val)
    loss = -f1_score(y_val, y_pred, average='macro')

    return {'loss': loss, 'status': STATUS_OK}

# Define the search space for hyperparameters
space = {
    'model__n_estimators': hp.choice('model__n_estimators', range(50, 500)),
    'model__learning_rate': hp.quniform('model__learning_rate', 0.01, 0.2, 0.01),
    'model__max_depth': hp.choice('model__max_depth', range(3, 14)),
    'model__min_child_weight': hp.choice('model__min_child_weight', range(1, 10)),
    'model__gamma': hp.uniform('model__gamma', 0.0, 0.5),
    'model__subsample': hp.uniform('model__subsample', 0.5, 1.0),
    'model__colsample_bytree': hp.uniform('model__colsample_bytree', 0.5, 1.0),
    'model__reg_alpha': hp.uniform('model__reg_alpha', 0.0, 1.0),
    'model__reg_lambda': hp.uniform('model__reg_lambda', 1.0, 4.0),
    'model__scale_pos_weight': hp.uniform('model__scale_pos_weight', 1.0, 10.0),
    'model__max_delta_step': hp.choice('model__max_delta_step', range(1, 10)),
}
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

# Train the final pipeline on the full dataset with the best parameters
model_pipeline.set_params(**best_hyperparams)
model_pipeline.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

y_pred = model_pipeline.predict(X_val)

#%%
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print('1. The F-1 score of the model {}\n'.format(f1_score(y_val, y_pred, average='macro')))
print('2. The recall score of the model {}\n'.format(recall_score(y_val, y_pred, average='macro')))
print('3. Classification report \n {} \n'.format(classification_report(y_val, y_pred)))
print('4. Confusion matrix \n {} \n'.format(confusion_matrix(y_val, y_pred)))
# %%

data['is goal'].value_counts()/len(data)

# %%

# After fitting the pipeline with the best parameters...
# Access the XGBClassifier from the pipeline
xgb_model = model_pipeline.named_steps['model']

# Get feature importances
feature_importances = xgb_model.feature_importances_

# Access the OneHotEncoder from the pipeline
onehot_encoder = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']

# Access the categorical feature names after one-hot encoding
categorical_feature_names = onehot_encoder.get_feature_names_out(input_features=categorical_cols)

# Combine with numerical feature names to create the full list
full_feature_names = list(numerical_cols) + list(categorical_feature_names) + ['year', 'month', 'day', 'period time']

#%%

full_feature_names

#%%

# Create a DataFrame to view feature names and their importance scores
importances = pd.DataFrame({'feature': full_feature_names, 'importance': feature_importances})

# Sort the DataFrame to see the most important features at the top
importances = importances.sort_values(by='importance', ascending=False)

# Display the feature importances
print(importances)



# %%