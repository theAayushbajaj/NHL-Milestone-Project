import json
import requests
import pandas as pd
import logging


logger = logging.getLogger(_name_)


class ServingClient:
    def _init_(self, ip: str = "0.0.0.0", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features

        # any other potential initialization

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """
        url = f"{self.base_url}/predict"
        data = X.to_json(orient="split")
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, data=data, headers=headers)

        if response.status_code == 200:
            return pd.DataFrame(response.json())
        else:
            logger.error(f"Failed to predict: {response.text}")
            return pd.DataFrame()
        # raise NotImplementedError("TODO: implement this function")

    def logs(self) -> dict:
        """Get server logs"""
        url = f"{self.base_url}/logs"
        response = requests.get(url)

        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to get logs: {response.text}")
            return {}
        # raise NotImplementedError("TODO: implement this function")

    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """
        url = f"{self.base_url}/download_registry_model"
        data = {"workspace": workspace, "model": model, "version": version}
        response = requests.post(url, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to download registry model: {response.text}")
            return {}
        # raise NotImplementedError("TODO: implement this function")