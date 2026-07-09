import requests

class GARCH:
    def __init__(self, data):
        self.data = data

    def fit(self):
        # Fit the GARCH model to the data
        pass

    def predict(self, steps) -> float:
        ticker = "BYMA"
        returns = requests.post(f"https://financiar186--garch-model-main-dev.modal.run?ticker={ticker}")
        return returns.json().get("prediction", [0])