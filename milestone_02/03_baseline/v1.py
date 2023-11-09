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

#%%

## import comet_ml at the top of your file
my_key = 'IGFjJ2mP1ZZPVHurdGupI5DJt'

# Create an experiment with your api key
experiment = Experiment(
    api_key=my_key,
    project_name="baseline model",
    workspace="2nd milestone",
)

#%%

## Create an experiment with your api key
experiment = Experiment(
    api_key=my_key,
    project_name="baseline model",
    workspace="2nd milestone",
)

#%%
warnings.filterwarnings('ignore')

# load data
data = pd.read_csv('baseline_model_data.csv')
#%%

# season 16 to 19 as training data
train = data[data['season'] < 2020]
# season 20 as test data
test = data[data['season'] == 2020]

# validation set as last year of training
val_index = train['season'] == 2019
val = train[val_index]
train = train[~val_index]
#%%

# train a logistic regression model
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
# only use `shot distance`
features = ['shot distance']
# features = ['shot distance', 'shot angle', 'empty net']
target = ['is goal']
clf.fit(train[features], train[target])

#%%

# evaluate the model with a classification report
from sklearn.metrics import classification_report
print(classification_report(val[target], clf.predict(val[features])))    

# evaluate the model with a confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(val[target], clf.predict(val[features])))

#%%

# Compute confusion matrix
cm = confusion_matrix(val[target], clf.predict(val[features]))
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


#%%


# plot a confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(confusion_matrix(val[target], clf.predict(val[features])), annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#%%

# create a function that plots these last 4 plots

def plot_calibration_curve(model, features, target, val, train, experiment):
    # initialize the model
    model = model()

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
features = ['shot distance']
experiment1 = Experiment(api_key=my_key, project_name="baseline model", workspace="2nd milestone")
plot_calibration_curve(LogisticRegression, features, target, val, train, experiment1)
experiment1.end()
#%%
features = ['shot angle']
experiment2 = Experiment(api_key=my_key, project_name="baseline model", workspace="2nd milestone")
plot_calibration_curve(LogisticRegression, features, target, val, train, experiment2)
experiment2.end()
#%%
features = ['shot angle', 'shot distance']
experiment3 = Experiment(api_key=my_key, project_name="baseline model", workspace="2nd milestone")
plot_calibration_curve(LogisticRegression, features, target, val, train, experiment3)
experiment3.end()
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

experiment4 = Experiment(api_key=my_key, project_name="baseline model", workspace="2nd milestone")
plot_calibration_curve(RandomModel, features, target, val, train, experiment4)
experiment4.end()


# %%
