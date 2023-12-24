import json
import requests
import pandas as pd
import logging

log = logging.getLogger(__name__)

class ModelClient:
    def __init__(self, host: str = "127.0.0.1", server_port: int = 5000, selected_features=None):
        self.api_url = f"http://{host}:{server_port}"
        log.info(f"Client setup complete; API URL: {self.api_url}")

        if selected_features is None:
            selected_features = ["distance"]
        self.selected_features = selected_features

        # Additional initializations if needed

    def request_prediction(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the data for a POST request to obtain predictions. Sends the request to the
        server and converts the response back into a DataFrame matching the input DataFrame's index.
        
        Args:
            input_df (DataFrame): The DataFrame to be sent to the prediction server.
        """
        prediction_url = f"{self.api_url}/predict"
        payload = input_df.to_json(orient="split")
        headers = {'Content-Type': 'application/json'}
        server_response = requests.post(prediction_url, data=payload, headers=headers)

        if server_response.status_code == 200:
            return pd.DataFrame(server_response.json())
        else:
            log.error(f"Prediction request failed: {server_response.text}")
            return pd.DataFrame()

    def retrieve_logs(self) -> dict:
        """Fetches logs from the server"""
        logs_url = f"{self.api_url}/logs"
        server_response = requests.get(logs_url)

        if server_response.status_code == 200:
            return server_response.json()
        else:
            log.error(f"Log retrieval failed: {server_response.text}")
            return {}

    def fetch_model_from_registry(self, workspace_name: str, model_name: str, model_version: str) -> dict:
        """
        Requests the server to download a specific model version from a model registry. 

        Args:
            workspace_name (str): The workspace containing the model in the registry
            model_name (str): The name of the model in the registry
            model_version (str): The specific version of the model to download
        """
        model_url = f"{self.api_url}/download_registry_model"
        request_data = {"workspace": workspace_name, "model": model_name, "version": model_version}
        server_response = requests.post(model_url, json=request_data)

        if server_response.status_code == 200:
            return server_response.json()
        else:
            log.error(f"Model download request failed: {server_response.text}")
            return {}
