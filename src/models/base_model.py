class BaseModelWrapper:
    def fit(self, dataset):
        return self

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass
