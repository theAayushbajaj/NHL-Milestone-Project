from utils import dataload
import requests

X_test = dataload("df_feature_engineering.csv")

request = {"workspace": "2nd-milestone","registry_name": "03_baseline_models_question1","model": "03_baseline_models_question1.pkl","version": "1.0.0"}
# r = requests.post("http://127.0.0.1:6060/download_registry_model",json=request)
r = requests.post("http://127.0.0.1:6060/predict", json=X_test.to_json())

print(r.status_code)
print(r.json())
