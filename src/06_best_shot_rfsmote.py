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


# Random Forest with SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from hyperopt import hp, fmin, tpe, Trials, space_eval
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

#%%
# get api key from text file
COMET_API_KEY = open('comet_api_key.txt').read().strip()

# Create an experiment with your api key
experiment = Experiment(
    api_key=COMET_API_KEY,
    project_name="Best Shot model 1",
    workspace="2nd milestone",
    log_code=True
)
experiment.log_code(file_name='06_best_shot.py')
#%%
def advanced_question1(experiment):
    '''
    . Train an XGBoost classifier using the same dataset using only the distance and angle features (similar to part 3). 
    Don’t worry about hyperparameter tuning yet, this will just serve as a comparison to the baseline before we add more features. 
    Add the corresponding curves to the four figures in your blog post. Briefly (few sentences) discuss your training/validation 
    setup, and compare the results to the Logistic Regression baseline. Include a link to the relevant comet.ml entry for this experiment, 
    but you do not need to log this model to the model registry.
    '''
    #%%
    data_baseline = pd.read_csv('data/baseline_model_data.csv')
    train_base, val_base, test_base = utils.split_train_val_test(data_baseline)

    #%%

    X_train = train_base[['shot_distance', 'shot_angle']]
    y_train = train_base['is_goal']
    X_val = val_base[['shot_distance', 'shot_angle']]
    y_val = val_base['is_goal']

    #%%

    features = ['shot_distance', 'shot_angle']
    target = ['is_goal']
    #%%
    # Define the pipeline with SMOTE and Random Forest
    rf_smote_pipeline = ImbPipeline(steps=[
        ('scaler', MinMaxScaler()),
        ('smote', SMOTE(random_state=42)),  # Apply SMOTE
        ('classifier', RandomForestClassifier(random_state=42))  # Random Forest Classifier
    ])

    # Fit the pipeline
    rf_smote_pipeline.fit(X_train, y_train)

    #%%
    model_reg_filename = f"bestshot_model_01.pkl"  # Modify as needed
    # Now call the function with this pipeline
    utils.plot_calibration_curve(model = rf_smote_pipeline, 
                                 features = features, 
                                 target = target, 
                                 val = val_base, 
                                 train = train_base, 
                                 model_reg_filename = model_reg_filename,
                                 tags = ["rfSMOTE_model_baseline", "calibration_curve"], 
                                 experiment = experiment,
                                 legend='RfSMOTE')

#%%
def preprocess_question2(data):
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

    # Categorical columns and corresponding transformers
    categorical_cols = X_train.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()

    # Numerical columns and corresponding transformers
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_transformer = Pipeline(steps=[
        #('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    # We need to convert booleans to integers before one-hot encoding
    for col in categorical_cols:
        if X_train[col].dtype == 'bool':
            X_train[col] = X_train[col].astype(int)
            X_val[col] = X_val[col].astype(int)

    print(categorical_cols)
    categorical_transformer = Pipeline(steps=[
        #('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Add the custom transformers to the preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'  # This drops the columns that we haven't transformed
    )

    # Create the preprocessing and modeling pipeline with Random Forest
    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', RandomForestClassifier(random_state=42))
    ])

    return model_pipeline, X_train, y_train, X_val, y_val

#%%
def hyperparameter_tuning_question2(model_pipeline, X_train, y_train, X_val, y_val):
    def objective(params):
        # Update only the model's parameters
        model_pipeline.named_steps['model'].set_params(**params)
        model_pipeline.fit(X_train, y_train)

        y_pred = model_pipeline.predict(X_val)
        loss = -f1_score(y_val, y_pred, average='macro')

        return {'loss': loss, 'status': STATUS_OK}

    # Define the search space for RandomForest hyperparameters
    space = {
        'n_estimators': hp.choice('n_estimators', range(50, 500)),
        'max_depth': hp.choice('max_depth', range(3, 14)),
        'min_samples_split': hp.choice('min_samples_split', range(2, 10)),
        'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 5)),
        'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
    }
    # Initialize Trials object to keep track of results
    trials = Trials()

    # Run the optimization
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=10,
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
def advanced_question2():
    #%%
    data_fe2 = pd.read_csv('data/new_data_for_modeling_tasks/df_data.csv') 
    #%%

    # Preprocess the data and get the pipeline
    model_pipeline, X_train, y_train, X_val, y_val = preprocess_question2(data_fe2)

    # Dropping NaN values to ensure data quality
    X_train = X_train.dropna()
    y_train = y_train[X_train.index]
    X_val = X_val.dropna()
    y_val = y_val[X_val.index]
    #%%

    # Perform hyperparameter tuning
    best_hyperparams = hyperparameter_tuning_question2(model_pipeline, X_train, y_train, X_val, y_val)

    #%%


    # Set the best hyperparameters to the model in the pipeline
    model_pipeline.named_steps['model'].set_params(**best_hyperparams)

    # Train the model with the best hyperparameters
    model_pipeline.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    #%%

    # Plot calibration curve (make sure the function is adapted for Random Forest)
    model_reg_filename = f"advanced_question2_rf_model.pkl"
    utils.plot_calibration_curve(model=model_pipeline, 
                                 features=X_train.columns, 
                                 target=['is_goal'], 
                                 val=pd.concat([X_val, y_val], axis=1), 
                                 train=X_train, 
                                 model_reg_filename=model_reg_filename,
                                 tags=["RandomForest_model_allFeatures", "calibration_curve"], 
                                 experiment=experiment,
                                 legend='RandomForest')
    #%%
    return model_pipeline


    #%%
    experiment.end()
# %%