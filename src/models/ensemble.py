class EnsembleModel:
    def __init__(self, models: list):
        self.models = models

    def predict(self, features):
        return sum(model.predict(features) for model in self.models) / len(self.models)
