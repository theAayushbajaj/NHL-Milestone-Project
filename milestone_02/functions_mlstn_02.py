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
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
#from sklearn.metrics import calibration_curve, roc_curve
import pandas as pd

#%%

def split_train_val_test(data):
    """
    Splits the input dataframe into training, validation, and test sets based on the season column.

    Args:
    - df_data: pandas DataFrame containing the data to be split

    Returns:
    - train: pandas DataFrame containing the training data (seasons 16-19)
    - val: pandas DataFrame containing the validation data (season 19)
    - test: pandas DataFrame containing the test data (season 20)
    """
    # season 16 to 19 as training data
    train = data[data['season'] < 2020]
    # season 20 as test data
    test = data[data['season'] == 2020]

    # validation set as last year of training
    val_index = train['season'] == 2019
    val = train[val_index]
    train = train[~val_index]
    return train, val, test


# %%

def plot_calibration_curve(model, features, target, val, train, experiment):
    # initialize the model
    #model = model()

    # fit the model
    model.fit(train[features], train[target])

    # Log the model to the experiment
    model_filename = f"model_{features[0]}.pkl"  # Modify as needed
    joblib.dump(model, model_filename)
    experiment.log_model(model_filename, model_filename)

    # Add tags to the experiment
    tags = [f"model_{features[0]}", "calibration_curve"]
    experiment.add_tags(tags)

    # get the probability of the prediction
    val['prob'] = model.predict_proba(val[features])[:, 1]

    # Add tags to the experiment
    tags = [f"model_{features[0]}", "calibration_curve"]
    experiment.add_tags(tags)

    # get the probability of the prediction
    val['prob'] = model.predict_proba(val[features])[:, 1]
    
    # get the goal rate
    goal_rate = val.groupby(pd.qcut(val['prob'], 10))['is goal'].mean()
    
    # convert interval index to its midpoint values
    goal_rate.index = goal_rate.index.map(lambda x: x.mid)
    
    # get the cumulative sum of goals
    cumulative_goals = val.groupby(pd.qcut(val['prob'], 10))['is goal'].sum().cumsum()
    
    # convert to proportion
    cumulative_goals = cumulative_goals / cumulative_goals.max()
    
    # convert interval index to its midpoint values
    cumulative_goals.index = cumulative_goals.index.map(lambda x: x.mid)
    
    # get the reliability diagram
    prob_true, prob_pred = calibration_curve(val[target], model.predict_proba(val[features])[:, 1], n_bins=10)
    
    # plot and save the goal rate plot
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.plot(goal_rate)
    plt.xlabel('Model Probability Percentile')
    plt.ylabel('Goal Rate')
    goal_rate_filename = f"goal_rate_{features[0]}.png"
    plt.savefig(goal_rate_filename)
    experiment.log_image(goal_rate_filename)
    
    # plot and save the cumulative goals plot
    plt.subplot(2, 2, 2)
    plt.plot(cumulative_goals)
    plt.xlabel('Model Probability Percentile')
    plt.ylabel('Cumulative Goals')
    cumulative_goals_filename = f"cumulative_goals_{features[0]}.png"
    plt.savefig(cumulative_goals_filename)
    experiment.log_image(cumulative_goals_filename)
    
    # plot the ROC curve
    plt.subplot(2, 2, 3)
    fpr, tpr, thresholds = roc_curve(val[target], model.predict_proba(val[features])[:, 1])
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    roc_curve_filename = f"roc_curve_{features[0]}.png"
    plt.savefig(roc_curve_filename)
    experiment.log_image(roc_curve_filename)
    
    # plot the reliability diagram
    plt.subplot(2, 2, 4)
    plt.plot(prob_pred, prob_true)
    plt.xlabel('Model Probability')
    plt.ylabel('Actual Probability')
    reliability_diagram_filename = f"reliability_diagram_{features[0]}.png"
    plt.savefig(reliability_diagram_filename)
    experiment.log_image(reliability_diagram_filename)
    
    plt.tight_layout()
    plt.show()

#%%