import numpy as np
import pandas as pd
import warnings
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from comet_ml import Experiment
import joblib
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, recall_score

def plot_confusion_matrix(y_true, y_pred):
    """
    Plots the confusion matrix and saves it as an image.
    
    Args:
    y_true (array): True labels
    y_pred (array): Predicted labels
    class_names (list): Names of the classes
    filename (str): Filename for the saved plot
    """
    confusion_matrix_filename = 'confusion_matrix.png'
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')

    plt.savefig(confusion_matrix_filename)
    return confusion_matrix_filename

def plot_classification_report(y_true, y_pred):
    """
    Plots the classification report as a heatmap and saves it as an image.

    Args:
    y_true (array): True labels
    y_pred (array): Predicted labels
    """
    classification_report_filename = 'classification_report.png'
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    # Remove the 'support' column to avoid large numbers affecting the colormap intensity
    sns.set(font_scale=1.2)  # Adjust to fit
    plt.figure(figsize=(10, 6))  # Adjust to fit
    heatmap = sns.heatmap(df_report.iloc[:-1, :-1], annot=True, fmt=".2f", cmap='Blues', cbar=False, linewidths=1, linecolor='black')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    
    plt.title('Classification Report', pad=20)
    plt.ylabel('Class Labels')
    plt.xlabel('Metrics')
    plt.tight_layout()  # Adjust the layout to fit the figure size
    plt.savefig(classification_report_filename)
    plt.close()  # Close the plot to free memory
    return classification_report_filename

def draw_missing_data_table(df):
    '''
    Docstring: Returns a datarframe with percent of missing/nan values per feature/column
    
    Parameters:
    ------------
    df: dataframe object
    
    Returns:
    ------------
    Dataframe containing missing value information
    '''
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent of NaNs'])
    return missing_data

class StratifiedValidationClass(object):
    def __init__(self, n_splits, base_models):
        self.n_splits = n_splits
        self.base_models = base_models

    def predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        no_class = len(np.unique(y))

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, 
                                     random_state = 42).split(X, y))

        train_proba = np.zeros((X.shape[0], no_class))
        test_proba = np.zeros((T.shape[0], no_class))
        
        train_pred = np.zeros((X.shape[0], len(self.base_models)))
        test_pred = np.zeros((T.shape[0], len(self.base_models)* self.n_splits))
        f1_scores = np.zeros((len(self.base_models), self.n_splits))
        recall_scores = np.zeros((len(self.base_models), self.n_splits))
        
        test_col = 0
        for i, clf in enumerate(self.base_models):
            
            for j, (train_idx, valid_idx) in enumerate(folds):
                
                X_train = X[train_idx]
                Y_train = y[train_idx]
                X_valid = X[valid_idx]
                Y_valid = y[valid_idx]
                
                clf.fit(X_train, Y_train)
                
                valid_pred = clf.predict(X_valid)
                recall  = recall_score(Y_valid, valid_pred, average='macro')
                f1 = f1_score(Y_valid, valid_pred, average='macro')
                
                recall_scores[i][j] = recall
                f1_scores[i][j] = f1
                
                train_pred[valid_idx, i] = valid_pred
                test_pred[:, test_col] = clf.predict(T)
                test_col += 1
                
                ## Probabilities
                valid_proba = clf.predict_proba(X_valid)
                train_proba[valid_idx, :] = valid_proba
                test_proba  += clf.predict_proba(T)
                
                print( "Model- {} and CV- {} recall: {}, f1_score: {}".format(i, j, recall, f1))
                
            test_proba /= self.n_splits
            
        return train_proba, test_proba, train_pred, test_pred
    
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
    goal_rate = val.groupby(pd.qcut(val['prob'], 10))['is_goal'].mean()
    
    # convert interval index to its midpoint values
    goal_rate.index = goal_rate.index.map(lambda x: x.mid)
    
    # get the cumulative sum of goals
    cumulative_goals = val.groupby(pd.qcut(val['prob'], 10))['is_goal'].sum().cumsum()
    
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
