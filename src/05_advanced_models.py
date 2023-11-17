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
    # Define the pipeline
    xg_pipeline = Pipeline(steps=[
        ('scaler', MinMaxScaler()),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ])

    # Fit the pipeline
    xg_pipeline.fit(X_train, y_train)

    #%%
    model_reg_filename = f"advanced_question1_model.pkl"  # Modify as needed
    # Now call the function with this pipeline
    utils.plot_calibration_curve(model = xg_pipeline, 
                                 features = features, 
                                 target = target, 
                                 val = val_base, 
                                 train = train_base, 
                                 model_reg_filename = model_reg_filename,
                                 tags = ["Advanced  Q1","XGBoost Baseline", "calibration_curve"], 
                                 experiment = experiment,
                                 legend = 'XGBoost Baseline')

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
def hyperparameter_tuning_question2(model, X_train, y_train, X_val, y_val):
    def objective(params):
        model.set_params(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        loss = -f1_score(y_val, y_pred, average='macro')

        return {'loss': loss, 'status': STATUS_OK}

    # Define the search space for hyperparameters
    space = {
        'n_estimators': hp.choice('n_estimators', range(50, 500)),
        'learning_rate': hp.quniform('learning_rate', 0.01, 0.2, 0.01),
        'max_depth': hp.choice('max_depth', range(3, 14)),
        'min_child_weight': hp.choice('min_child_weight', range(1, 10)),
        'gamma': hp.uniform('gamma', 0.0, 0.5),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
        'reg_lambda': hp.uniform('reg_lambda', 1.0, 4.0),
        'scale_pos_weight': hp.uniform('scale_pos_weight', 1.0, 10.0),
        'max_delta_step': hp.choice('max_delta_step', range(1, 10)),
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
    X_train, y_train, X_val, y_val, _, _ = preprocess(data_fe2)
    #%%
    # Perform hyperparameter tuning
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    best_hyperparams = hyperparameter_tuning_question2(model,X_train, y_train, X_val, y_val)
    #%%
    # Train the model with the best hyperparameters
    model.set_params(**best_hyperparams)
    model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    #%%
    # Plot calibration curve
    model_reg_filename = f"advanced_question2_model.pkl"
    utils.plot_calibration_curve(model = model, 
                                 features = X_train.columns, 
                                 target = ['is_goal'], 
                                 val = pd.concat([X_val,y_val],axis=1), 
                                 train = X_train, 
                                 model_reg_filename = model_reg_filename,
                                 tags = ["Advanced Q2","XGBoost Tuned", "calibration_curve"], 
                                 experiment = experiment,
                                 legend = 'XGBoost Tuned')
    #%%
    return clf
#%%
def feature_selection_question3(model, X_train, X_val, y_train, y_val):
    # Fit the model pipeline with your training data
    model.fit(X_train, y_train)

    # Create a SHAP TreeExplainer using the fitted XGBClassifier model
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values - this can be compute-intensive for large datasets
    shap_values = explainer.shap_values(X_train, approximate=True)

    # Get the feature names after preprocessing
    feature_names = X_train.columns

    # Handle multi-class case
    if isinstance(shap_values, list):
        # Sum the absolute SHAP values across all output classes
        shap_sum = np.sum([np.abs(sv).mean(0) for sv in shap_values], axis=0)
    else:
        # For binary classification, take the mean of the absolute SHAP values
        shap_sum = np.abs(shap_values).mean(0)

    # Create a series of SHAP values
    shap_series = pd.Series(shap_sum, index=feature_names).sort_values(ascending=False)

    # Get the top 15 feature names and their indices
    top_features_indices = np.argsort(-shap_sum)[:15]
    top_feature_names = shap_series.index[:15].tolist()

    # Visualize the SHAP values with a beeswarm plot
    plt.switch_backend('Agg')
    shap.summary_plot(shap_values, X_train, plot_type='dot', feature_names=feature_names)
    fig = plt.gcf()
    fig.savefig('shap_beeswarm_plot.png', bbox_inches='tight')
    plt.close(fig)

    # Log the SHAP summary plot to Comet
    experiment.log_image('shap_beeswarm_plot.png', name='SHAP Beeswarm Plot')

    return top_feature_names, top_features_indices

#%%
def advanced_question3():
    #%%
    data_fe2 = pd.read_csv('data/new_data_for_modeling_tasks/df_data.csv') 
    X_train, y_train, X_val, y_val, _, _ = preprocess(data_fe2)
    #%%
    # Perform hyperparameter tuning
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    best_hyperparams = hyperparameter_tuning_question2(model,X_train, y_train, X_val, y_val)
    #%%
    model.set_params(**best_hyperparams)
    top_feature_names, top_feature_indices = feature_selection_question3(model, X_train, X_val, y_train, y_val)
    #%%
    X_train = X_train[top_feature_names]
    X_val = X_val[top_feature_names]

    best_hyperparams = hyperparameter_tuning_question2(model,X_train, y_train, X_val, y_val)
    model.set_params(**best_hyperparams)
    model.fit(pd.concat([X_train,X_val]), pd.concat([y_train,y_val]))

    #%%
    model_reg_filename = f"advanced_question3_model.pkl"
    utils.plot_calibration_curve(model = model, 
                                 features = X_train.columns, 
                                 target = ['is_goal'], 
                                 val = pd.concat([X_val,y_val],axis=1), 
                                 train = X_train, 
                                 model_reg_filename = model_reg_filename,
                                 tags = ["Advanced Q3","Tuned XGBoost on SHAP", "calibration_curve"], 
                                 experiment = experiment,
                                 legend='XGBoost Tuned on SHAP')
    #%%
    return clf

if __name__ == '__main__':
    #advanced_question1()
    clf = advanced_question2()
    #advanced_question3(clf)

    #%%
    experiment.end()
# %%
