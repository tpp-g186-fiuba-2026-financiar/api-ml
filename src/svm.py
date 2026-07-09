import requests

class SupportVectorMachine:
    def __init__(self, data):
        self.data = data

    def fit(self):
        # Fit the SVM model to the data
        pass

    def predict(self, steps) -> str:
        ticker = "BYMA"
        returns = requests.post(f"https://financiar186--svm-model-main-dev.modal.run?ticker={ticker}")
        return returns.json().get("prediction", [0])