from comet_ml import Experiment
import os 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, \
precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import json 


def renamer(df): 
    df = df.copy() 
    cols = df.columns.to_list()
    cols_space = [col.replace(' ', '_') for col in cols] 
    cols_rename = {cols[i]: cols_space[i] for i in range(len(cols))} 
    return cols_rename 

def ExperimentTrack(model_path, parameters_log, metrics_log, hyperparameters_log, data, project_name, workspace): 
    my_key = 'IGFjJ2mP1ZZPVHurdGupI5DJt'
    print('Experiment Started!') 
    # Create an experiment with your api key
    experiment = Experiment(
    api_key=my_key,
    project_name="baseline model",
    workspace="2nd milestone") 

    experiment.log_dataset_hash(data)
    experiment.log_parameters(parameters_log)
    experiment.log_metrics(metrics_log)
    experiment.log_model(model_path)
    experimet.end() 
    print('Experiment Ended!') 
    pass 

def SaveSklearnModel(model_object, load=False): 
    model = pickle.dumps(model_object) 
    print('Model saved!') 
    if load:
        model_loaded = pickle.loads(model)
        return model_loaded 
    else:
        pass 
        
def SaveXGBoostModel(model_object, model_name, load=False):
    model_name = model_name + '.json'
    model_object.save_model(model_name)
    model_object.save_config()
    print(f'Model with name {model_name} saved!')
    if load: 
        xgb_model = xgboost.XGBClassifier()
        loaded_model = xgb_model.load_model(model_name)
        config = model_object.load_config()
        return loaded_model, config 
    else: 
        pass
    
    
    

if __name__== 'main': 
    pass