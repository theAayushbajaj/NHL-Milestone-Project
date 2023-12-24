"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
from pathlib import Path
import logging

from waitress import serve
from flask import Flask, jsonify, request, abort
from flask import current_app
import sklearn
import pandas as pd
import joblib

import ift6758
from comet_ml import API
import pickle
import numpy as np
import json


LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
MODELS_DIR = "../models"
version = '1.0.0'
app = Flask(__name__)
COMET_API_KEY = os.environ.get('COMET_API_KEY')
logging.basicConfig(filename=LOG_FILE, level=logging.INFO)
logger = logging.getLogger('waitress')
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_FILE)
fh.setLevel(logging.INFO)
logger.addHandler(fh)
model = None

@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    global model 

    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

    workspace_name = "2nd-milestone"
    default_registry = "03_baseline_models_question1_v2"
    default_model = "03_baseline_models_question2_v2"
    default_model_dir = os.path.join(MODELS_DIR, default_model)
    request = {
        "workspace": workspace_name,
        "registry_name": default_registry,
        "version": "1.0.0",
    }

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    if (not os.path.isfile(default_model_dir)):
        app.logger.info(f"Downloading the default model {default_model} fom CometML")
        API(api_key=COMET_API_KEY).download_registry_model(**request, output_path=MODELS_DIR)

        if (not os.path.isfile(default_model_dir)):
            app.logger.info("Cannot download the model. Check the comet project and API key.")
        else:
            app.logger.info("Downloaded the model succesfully.")
            model = joblib.load(default_model_dir)
    else:
        model = joblib.load(default_model_dir)
        app.logger.info(f"The default model {default_model} already exist. Load default model.")

    from sklearn.base import is_classifier
    if not is_classifier(model):
        print("Loaded object is not a scikit-learn classifier")

    app.model = model
    app.logger.info(f'App model: {app.model}')
    app.model_name = default_model



@app.route("/")
def ping():
    message = f'NHL Analytics ({version}) is Active!!'
    app.logger.info(message)
    return message

@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""

    # TODO: read the log file specified and return the data
    response = {}
    with open(LOG_FILE) as f:
        for line in f:
            response[line] = line

    return jsonify(response)  # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }

    """
    # Get POST json data
    global model
    json = request.get_json()
    model_name = json['model']
    app.logger.info(json)

    # Check if the requested model is already loaded
    if os.path.isfile(os.path.join(MODELS_DIR, model_name)):
        app.logger.info(f'Model {json["model"]} is already downloaded.')
        model = joblib.load(os.path.join(MODELS_DIR, model_name))
        model_loaded = True
    else:
        # Existing logic to download and load the model
        app.logger.info(f'Downloading model {json["model"]} from comet.ml')
        #api = API(api_key=COMET_API_KEY)
        try:
            API(api_key=COMET_API_KEY).download_registry_model(workspace=json['workspace'],
                                        registry_name=json['model'],
                                        version=json['version'],
                                        output_path="../models",
                                        expand=True)
            app.logger.info(f'Model {json["model"]} downloaded successfully')
            #model = pickle.load(open(os.path.join(MODELS_DIR, model_name), "rb"))
            model = joblib.load(os.path.join(MODELS_DIR, model_name))
            app.logger.info('Model loaded')
        except Exception as e:
            app.logger.info(f'Exception: {e}, Using default model')

        model_loaded = True

    response = json
    response['model_loaded'] = model_loaded
    app.logger.info(response)
    app.model = model
    app.model_name = json['model']
    return jsonify(response)  # response must be json serializable!

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here
    
@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    global model
    try:
        json_req = request.get_json()
        app.logger.info(json)

        df = pd.read_json(json_req)
        app.logger.info(f'First 5 rows of input: {df.head()}')

        y_pred = model.predict_proba(df)[:,1]
        app.logger.info(f'First 5 predictions {y_pred[:5]}')
        response = pd.DataFrame(y_pred).to_json()
        #app.logger.info(f'response sample {response}')

        app.logger.info(f'Number of predictions made: {y_pred.shape[0]}')
        unique, counts = np.unique(y_pred, return_counts=True)
        goal_percentage = counts[1]/y_pred.shape[0]
        app.logger.info(f'Goal percentage: {goal_percentage}, Number of Goals: {counts[1]}')

        app.logger.info(response)
        return jsonify(response)

    except Exception as e:
        app.logger.info(f"An error has occured: {e}")
        response = e
        return jsonify(response)

if __name__ == '__main__':
    app.run(port=6060, host="127.0.0.1", debug=True)

#serve(app, host='0.0.0.0', port=6060)







