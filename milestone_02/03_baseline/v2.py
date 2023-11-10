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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler



## import comet_ml at the top of your file
my_key = 'IGFjJ2mP1ZZPVHurdGupI5DJt'

# Create an experiment with your api key
experiment = Experiment(
    api_key=my_key,
    project_name="baseline model",
    workspace="2nd milestone",
)



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

# Define the features and target variable
features = ['shot distance']  # You can add more features here
target = ['is goal']  # Ensure the target is a string if it's a single column

# Create the pipeline
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),  # This will scale features to the [0, 1] range
    ('logistic_regression', LogisticRegression())
])

# Fit the pipeline to your data
pipeline.fit(train[features], train[target])

# Now you can use `pipeline` to make predictions, and the features will be automatically scaled

#%%

# evaluate the model with a classification report
from sklearn.metrics import classification_report
print(classification_report(val[target], pipeline.predict(val[features])))

# evaluate the model with a confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(val[target], pipeline.predict(val[features])))
#%%



#%%

# Compute confusion matrix
cm = confusion_matrix(val[target], pipeline.predict(val[features]))
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
sns.heatmap(confusion_matrix(val[target], pipeline.predict(val[features])), annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


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
os.chdir('./03_baseline')
print(os.getcwd())


#%%
features = ['shot distance']
experiment1 = Experiment(api_key=my_key, project_name="baseline model", workspace="2nd milestone")
_fct.plot_calibration_curve(pipeline, features, target, val, train, experiment1)
experiment1.end()
#%%
features = ['shot angle']
experiment2 = Experiment(api_key=my_key, project_name="baseline model", workspace="2nd milestone")
_fct.plot_calibration_curve(pipeline, features, target, val, train, experiment2)
experiment2.end()
#%%
features = ['shot angle', 'shot distance']
experiment3 = Experiment(api_key=my_key, project_name="baseline model", workspace="2nd milestone")
_fct.plot_calibration_curve(pipeline, features, target, val, train, experiment3)
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

# pipeline with random model
pipeline_random = Pipeline([
    ('scaler', MinMaxScaler()),  # This will scale features to the [0, 1] range
    ('random_model', RandomModel())
])

experiment4 = Experiment(api_key=my_key, project_name="baseline model", workspace="2nd milestone")
_fct.plot_calibration_curve(pipeline_random, features, target, val, train, experiment4)
experiment4.end()


# %%
