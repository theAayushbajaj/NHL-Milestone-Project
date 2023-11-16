#%%
# Basic libraries
import numpy as np
import pandas as pd

# Comet_mel to load models
from comet_ml import API

# Utilities
import utils

# Joblib for model persistence
import joblib
#%%
# get api key from text file
COMET_API_KEY = open('comet_api_key.txt').read().strip()
api = API(rest_api_key=COMET_API_KEY)

#%%

api.download_registry_model("2nd-milestone", "distance_model", "1.0.0",
                            output_path="./", expand=True)
api.download_registry_model("2nd-milestone", "angle_model", "1.0.0",
                            output_path="./", expand=True)
api.download_registry_model("2nd-milestone", "distance_angle_model", "1.0.0",
                            output_path="./", expand=True)



#%%

def test_it_question1():
    #%%

    # 3 logistic models
    data_baseline = pd.read_csv('data/baseline_model_data.csv')
    train_base, val_base, test_base = utils.split_train_val_test(data_baseline)

    #%%
    # evaluate the models on the test set
    # load the models from disk
    distance_model = joblib.load('distance_model.pkl')
    angle_model = joblib.load('angle_model.pkl')
    distance_angle_model = joblib.load('distance_angle_model.pkl')

    #%%
    # get the predictions
    y_pred_distance = distance_model.predict(test_base[['shot_distance']])
    y_pred_angle = angle_model.predict(test_base[['shot_angle']])
    y_pred_distance_angle = distance_angle_model.predict(test_base[['shot_distance', 'shot_angle']])
    #%%

    




# %%
