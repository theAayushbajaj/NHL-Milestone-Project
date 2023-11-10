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
#%%
def advanced_question1(experiment):
    '''
    . Train an XGBoost classifier using the same dataset using only the distance and angle features (similar to part 3). 
    Don’t worry about hyperparameter tuning yet, this will just serve as a comparison to the baseline before we add more features. 
    Add the corresponding curves to the four figures in your blog post. Briefly (few sentences) discuss your training/validation 
    setup, and compare the results to the Logistic Regression baseline. Include a link to the relevant comet.ml entry for this experiment, 
    but you do not need to log this model to the model registry.
    '''

    data_baseline = pd.read_csv('data/baseline_model_data.csv')
    train_base, val_base, test_base = utils.split_train_val_test(data_baseline)

    X_train = train_base[['shot distance', 'shot angle']]
    y_train = train_base['is goal']

    features = ['shot_distance', 'shot_angle']
    target = ['is_goal']
    # Define the pipeline
    xg_pipeline = Pipeline(steps=[
        ('scaler', MinMaxScaler()),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ])


    # Now call the function with this pipeline
    utils.plot_calibration_curve(xg_pipeline, features, target, val_base, train_base, experiment)

#%%
def preprocess_question2(data):
    class CustomFunctionTransformer(FunctionTransformer):
        def __init__(self, func, feature_names_out, validate=False):
            super().__init__(func, validate=validate)
            self.feature_names_out = feature_names_out

        def get_feature_names_out(self, input_features=None):
            return self.feature_names_out
    
    def fix_strength(df):
        strength = 'even'
        if df['num player home'] > df['num player away']:
            strength = 'power_play' if df['team shot'] == df['home team'] else 'short_handed'
        elif df['num player home'] < df['num player away']:
            strength = 'short_handed' if df['team shot'] == df['home team'] else 'power_play'
        df['strength'] = strength
        return df

    data = data.apply(fix_strength, axis=1)
    # split the data
    train, val, test = utils.split_train_val_test(data)
    X_train = train.drop(columns=['is goal'])
    y_train = train['is goal']
    X_val = val.drop(columns=['is goal'])
    y_val = val['is goal']

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
    date_transformer = CustomFunctionTransformer(
                        parse_game_date,
                        feature_names_out=['game_year', 'game_month', 'game_day']
                    )
    time_transformer = CustomFunctionTransformer(
                        parse_period_time,
                        feature_names_out=['game_year', 'game_month', 'game_day']
                    )


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

    return model_pipeline, X_train, y_train, X_val, y_val

#%%
def hyperparameter_tuning_question2(model, X_train, y_train, X_val, y_val):
    def objective(params):
        # Create the RandomForestClassifier with the given hyperparameters
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
    data_fe2 = pd.read_csv('data/data_for_remaining_tasks/df_data.csv') 
    model, X_train, y_train, X_val, y_val = preprocess_question2(data_fe2)
    #%%
    # Perform hyperparameter tuning
    best_hyperparams = hyperparameter_tuning_question2(model,X_train, y_train, X_val, y_val)
    #%%
    # Train the model with the best hyperparameters
    model.set_params(**best_hyperparams)
    model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    y_pred = model.predict(X_val)

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
    # Finally, log the model itself
    experiment.log_model("best_xgboost_model", model)
    #%%
    return clf
#%%
def feature_selection_question3(model, X_train, X_val, y_train, y_val):
    # Try lasso feature selection
    # Fit the model pipeline with your training data
    model.fit(X_train, y_train)

    # Extract the fitted XGBClassifier from the pipeline
    fitted_model = model.named_steps['model']

    # Create a SHAP TreeExplainer using the fitted XGBClassifier model
    explainer = shap.TreeExplainer(fitted_model)

    # Transform the features (X_train) using the preprocessor to get the transformed features
    X_train_transformed = model.named_steps['preprocessor'].transform(X_train)

    # Compute SHAP values - this can be compute-intensive for large datasets
    shap_values = explainer.shap_values(X_train_transformed)

    # Visualize the SHAP values (for example, a summary plot)
    shap.summary_plot(shap_values, X_train_transformed, plot_type='bar', show=False)
    plt.savefig('shap_summary_plot.png')

    # Log the SHAP summary plot to Comet
    experiment.log_image('shap_summary_plot.png', name='SHAP Summary Plot')

    # Return the top 20 features based on SHAP values
    # Aggregate the SHAP values across all output classes (important for multi-class classification)
    shap_sum = np.abs(shap_values).mean(axis=0)
    if isinstance(shap_sum, list):  # Handling multi-class outputs
        shap_sum = np.sum(np.array(shap_sum), axis=0)

    # Get the feature names after preprocessing
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()

    # Create a series of SHAP values
    shap_series = pd.Series(shap_sum, index=feature_names).sort_values(ascending=False)
    top_20_features = shap_series.head(20).index.tolist()

    # Transform features for X_train and X_val
    X_train_top_features = model.named_steps['preprocessor'].transform(X_train)[:, shap_series.head(20).index]
    X_val_top_features = model.named_steps['preprocessor'].transform(X_val)[:, shap_series.head(20).index]

    return X_train_top_features, X_val_top_features

    # # Fit the model pipeline with your training data
    # model.fit(X_train, y_train)
    # fitted_model = model.named_steps['model']

    # # Create a SHAP TreeExplainer using the fitted XGBClassifier model
    # explainer = shap.TreeExplainer(fitted_model)

    # # Transform the features (X_train) using the preprocessor to get the transformed features
    # X_train_transformed = model.named_steps['preprocessor'].transform(X_train)

    # # Compute SHAP values - this step can be compute-intensive for large datasets
    # shap_values = explainer.shap_values(X_train_transformed)

    # # Visualize the SHAP values (for example, a summary plot)
    # shap.summary_plot(shap_values, X_train_transformed, plot_type = 'bar',feature_names=X_train.columns, show=False)
    # plt.savefig('shap_summary_plot.png')

    # # Log the SHAP summary plot to Comet
    # experiment.log_image('shap_summary_plot.png', name='SHAP Summary Plot')

    # # return the top 10 features
    # shap_sum = np.abs(shap_values).mean(axis=0).mean(axis=0)
    # shap_series = pd.Series(shap_sum, index=X_train.columns).sort_values(ascending=False)
    # top_20_features = shap_series.head(20).index.tolist()
    
    # X_train = X_train[top_20_features]
    # X_val = X_val[top_20_features]

    # return X_train, X_val
#%%
def advanced_question3():
    #%%
    data_fe2 = pd.read_csv('data/data_for_remaining_tasks/df_data.csv') 
    model, X_train, y_train, X_val, y_val = preprocess_question2(data_fe2)
    #%%
    # Perform hyperparameter tuning
    best_hyperparams = hyperparameter_tuning_question2(model,X_train, y_train, X_val, y_val)
    
    #%%
    model.set_params(**best_hyperparams)
    X_train, X_val = feature_selection_question3(model, X_train, X_val, y_train, y_val)
    #%%
    # Train the model with the best hyperparameters
    model.set_params(**best_hyperparams)
    model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    y_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')
    recall = recall_score(y_val, y_pred, average='macro')
    experiment.log_metrics({"accuracy": accuracy, "f1_score": f1, "recall": recall})
    
    # Generate and log confusion matrix and classification report
    confusion_matrix_path = utils.plot_confusion_matrix(y_val, y_pred)
    classification_report_path = utils.plot_classification_report(y_val, y_pred)

    experiment.log_image(confusion_matrix_path, name='Confusion Matrix')
    experiment.log_image(classification_report_path, name='Classification Report')

    # Finally, log the model itself
    experiment.log_model("best_xgboost_model", clf)
    #%%
    return clf

if __name__ == '__main__':
    #advanced_question1()
    clf = advanced_question2()
    #advanced_question3(clf)
    experiment.end()