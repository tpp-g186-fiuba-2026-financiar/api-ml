import requests

class ARIMA:
    def __init__(self, data):
        self.data = data

    def fit(self):
        # Fit the ARIMA model to the data
        pass

    def predict(self, steps) -> float:
        ticker = "BYMA"
        returns = requests.post(f"https://financiar186--arima-model-main-dev.modal.run?ticker={ticker}&predictions=1&media_movil=10")
        return returns.json().get("predictions", [0])[0]
