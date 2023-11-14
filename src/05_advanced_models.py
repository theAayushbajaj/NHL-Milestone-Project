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
                                 tags = ["XGBoost_model_baseline", "calibration_curve"], 
                                 experiment = experiment,
                                 legend='q1')

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

    # Create the preprocessing and modeling pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ])

    return model_pipeline, X_train, y_train, X_val, y_val

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
    model, X_train, y_train, X_val, y_val = preprocess_question2(data_fe2)
    #%%
    # Perform hyperparameter tuning
    #best_hyperparams = hyperparameter_tuning_question2(model,X_train, y_train, X_val, y_val)
    best_hyperparams = {'model__colsample_bytree': 0.8027518366137031,
                        'model__gamma': 0.26323633422422865,
                        'model__learning_rate': 0.15,
                        'model__max_delta_step': 3,
                        'model__max_depth': 7,
                        'model__min_child_weight': 3,
                        'model__n_estimators': 239,
                        'model__reg_alpha': 0.6978729873816338,
                        'model__reg_lambda': 3.7646728674809875,
                        'model__scale_pos_weight': 3.4052468075491102,
                        'model__subsample': 0.7778593312583509}
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
                                 tags = ["XGBoost_model_allFeatures", "calibration_curve"], 
                                 experiment = experiment,
                                 legend='q2')
    #%%
    return clf
#%%
def feature_selection_question3(model, X_train, X_val, y_train, y_val):
    # Fit the model pipeline with your training data
    model.fit(X_train, y_train)
    
    # Extract the fitted XGBClassifier from the pipeline
    fitted_model = model.named_steps['model']
    
    # Create a SHAP TreeExplainer using the fitted XGBClassifier model
    explainer = shap.TreeExplainer(fitted_model)
    
    # Transform the features (X_train) using the preprocessor to get the transformed features
    X_train_transformed = model.named_steps['preprocessor'].transform(X_train)
    
    # Compute SHAP values - this can be compute-intensive for large datasets
    shap_values = explainer.shap_values(X_train_transformed, approximate=True)
    
    # Get the feature names after preprocessing
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    
    plt.switch_backend('Agg')

    # Your SHAP calculation code

    # Visualize the SHAP values with a beeswarm plot
    shap.summary_plot(shap_values, X_train_transformed, plot_type='dot', feature_names=feature_names)

    # Ensure that the figure is explicitly defined and saved
    fig = plt.gcf()  # Get the current figure before saving it
    fig.savefig('shap_beeswarm_plot.png', bbox_inches='tight')  # Use bbox_inches to include all plot elements
    plt.close(fig)  # Close the plot to free up memory
    # Log the SHAP summary plot to Comet
    experiment.log_image('shap_beeswarm_plot.png', name='SHAP Beeswarm Plot')

    # Return the top 20 features based on SHAP values
    # For multi-class classification, sum the SHAP values across all classes
    shap_sum = np.abs(shap_values).mean(axis=0)
    if isinstance(shap_sum, list):  # Multi-class case
        shap_sum = np.sum(np.array(shap_sum), axis=0)

    # Create a series of SHAP values
    shap_series = pd.Series(shap_sum, index=feature_names).sort_values(ascending=False)
    top_features_indices = np.argsort(-shap_sum)[:10]  # Get indices of top features

    # Select the top features for X_train and X_val
    X_val_transformed = model.named_steps['preprocessor'].transform(X_val)
    if isinstance(X_train_transformed, np.ndarray):
        X_train_top_features = X_train_transformed[:, top_features_indices]
    else:
        X_train_top_features = X_train_transformed.toarray()[:, top_features_indices]

    if isinstance(X_val_transformed, np.ndarray):
        X_val_top_features = X_val_transformed[:, top_features_indices]
    else:
        X_val_top_features = X_val_transformed.toarray()[:, top_features_indices]

    # Remove 'num__' and 'cat__' prefixes from the feature names
    def clean_feature_name(fname):
        return fname.replace('num__', '').replace('cat__', '')

    # Get clean top feature names
    top_feature_names = [clean_feature_name(feature_names[i]) for i in top_features_indices]
    return top_feature_names, top_features_indices


#%%
def advanced_question3():
    #%%
    data_fe2 = pd.read_csv('data/new_data_for_modeling_tasks/df_data.csv') 
    model_pipeline, X_train, y_train, X_val, y_val = preprocess_question2(data_fe2)
    #%%
    # Perform hyperparameter tuning
    #best_hyperparams = hyperparameter_tuning_question2(model_pipeline,X_train, y_train, X_val, y_val)
    best_hyperparams = {'model__colsample_bytree': 0.8443652326712037,
                        'model__gamma': 0.3836159341092055,
                        'model__learning_rate': 0.11,
                        'model__max_delta_step': 6,
                        'model__max_depth': 6,
                        'model__min_child_weight': 1,
                        'model__n_estimators': 385,
                        'model__reg_alpha': 0.4947013395389696,
                        'model__reg_lambda': 2.5789685513445617,
                        'model__scale_pos_weight': 3.956297703976836,
                        'model__subsample': 0.9275047750757387}
    
    #%%
    model_pipeline.set_params(**best_hyperparams)
    top_feature_names, top_feature_indices = feature_selection_question3(model_pipeline, X_train, X_val, y_train, y_val)
    #%%
    # Train the model with the best hyperparameters
    # First, fit and transform with the preprocessor
    preprocessor = clone(model_pipeline.named_steps['preprocessor'])
    X_train_transformed = preprocessor.fit_transform(X_train).toarray()

    # Now, fit the model separately with the transformed data
    model = clone(model_pipeline.named_steps['model'])
    X_train_transformed = X_train_transformed[:,top_feature_indices]
    X_val_transformed = preprocessor.transform(X_val).toarray()[:,top_feature_indices]
    model.fit(np.concatenate((X_train_transformed,X_val_transformed)), pd.concat([y_train,y_val]))

    #%%
    model_reg_filename = f"advanced_question3_model.pkl"
    utils.plot_calibration_curve(model = model_pipeline, 
                                 features = X_train.columns, 
                                 target = ['is_goal'], 
                                 val = pd.concat([X_val,y_val],axis=1), 
                                 train = X_train, 
                                 model_reg_filename = model_reg_filename,
                                 tags = ["Tuned_XGBoost_model_allFeatures", "calibration_curve"], 
                                 experiment = experiment,
                                 legend='q3')
    #%%
    return clf

if __name__ == '__main__':
    #advanced_question1()
    clf = advanced_question2()
    #advanced_question3(clf)

    #%%
    experiment.end()
# %%
