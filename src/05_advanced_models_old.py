import numpy as np
import pandas as pd
from comet_ml import Experiment

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, \
                                    StratifiedKFold

from sklearn.metrics import accuracy_score, classification_report
import utils
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

import utils
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval

def advanced_question1():
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
#%%
def log_experiment(experiment, best_params, best_score):
    # Log the best hyperparameters and best score to comet.ml
    experiment.log_parameters(best_params)
    experiment.log_metric("best_score", best_score)

#%%
def hyperparameter_tuning_question2(X_train, y_train, X_val, y_val, experiment):
    def objective(params):
        # Create the RandomForestClassifier with the given hyperparameters
        clf = XGBClassifier(**params)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)
        loss = -accuracy_score(y_val, y_pred)

        return {'loss': loss, 'status': STATUS_OK}

    # Define the search space for hyperparameters
    space = {
        'n_estimators': hp.choice('n_estimators', range(50, 500)),  # Number of gradient boosted trees
        'learning_rate': hp.quniform('learning_rate', 0.01, 0.2, 0.01),  # Boosting learning rate
        'max_depth': hp.choice('max_depth', range(3, 14)),  # Maximum tree depth for base learners
        'min_child_weight': hp.choice('min_child_weight', range(1, 10)),  # Minimum sum of instance weight(hessian) needed in a child
        'gamma': hp.uniform('gamma', 0.0, 0.5),  # Minimum loss reduction required to make a further partition on a leaf node of the tree
        'subsample': hp.uniform('subsample', 0.5, 1.0),  # Subsample ratio of the training instance
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),  # Subsample ratio of columns when constructing each tree
        'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),  # L1 regularization term on weights
        'reg_lambda': hp.uniform('reg_lambda', 1.0, 4.0),  # L2 regularization term on weights
        'scale_pos_weight': hp.uniform('scale_pos_weight', 1.0, 10.0),  # Balancing of positive and negative weights
        'max_delta_step': hp.choice('max_delta_step', range(1, 10)),  # Maximum delta step we allow each tree's weight estimation to be
        'objective': 'multi:softmax',  # Objective function for multiclass classification,
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
    log_experiment(experiment, best_hyperparams, best_score)

    return best_hyperparams

def advanced_question2():
    data_fe2 = pd.read_csv('data/data_for_remaining_tasks/df_data.csv')
    print(data_fe2.head())

    train, val, test = utils.split_train_val_test(data_fe2)

    X_train = train.drop(columns=['is goal'], axis=1)
    y_train = train['is goal']
    X_val = val.drop(columns=['is goal'], axis=1)
    y_val = val['is goal']

    # Perform hyperparameter tuning
    best_hyperparams = hyperparameter_tuning_question2(X_train, y_train, X_val, y_val, experiment)

    # Train the model with the best hyperparameters
    clf = XGBClassifier(**best_hyperparams)
    clf.fit(X_train, y_train)

    # Log the classifier's performance on the validation set
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    experiment.log_metric("val_accuracy", accuracy)

    # Finally, log the model itself
    experiment.log_model("best_xgboost_model", clf)

if __name__ == '__main__':
    # get api key from text file
    COMET_API_KEY = open('comet_api_key.txt').read().strip()

    # Create an experiment with your api key
    experiment = Experiment(
        api_key=COMET_API_KEY,
        project_name="Advanced model",
        workspace="2nd milestone",
    )
    advanced_question2()