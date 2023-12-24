from utils import dataload
import requests
import pandas as pd

X_test = pd.read_csv("baseline_model_data.csv")
X_test = X_test[['shot_angle', 'shot_distance']]
request = {"workspace": "2nd-milestone","registry_name": "03_baseline_models_question2_v2","model": "03_baseline_models_question2_v2","version": "1.0.0"}
r = requests.post("http://127.0.0.1:6060/download_registry_model",json=request)

r = requests.post("http://127.0.0.1:6060/predict", json=X_test.to_json())

print(r.status_code)
print(r.json())