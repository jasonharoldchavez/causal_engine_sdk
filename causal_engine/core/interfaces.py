import abc
import numpy as np

class AbstractCausalModel(abc.ABC):

    @abc.abstractmethod
    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        pass

    @abc.abstractmethod
    def predict(self, current_features: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def extract_causal_matrix(self) -> np.ndarray:
        pass
