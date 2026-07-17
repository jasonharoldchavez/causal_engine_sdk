import numpy as np
import scipy.linalg as la
from causal_engine.core.interfaces import AbstractCausalModel


class PluginLinearVAR(AbstractCausalModel):

    def __init__(self, num_signals, max_lags=3, ridge_lambda=1e-3):
        self.num_signals = num_signals
        self.max_lags = max_lags
        self.ridge_lambda = ridge_lambda
        self.B = None


    def fit(self, X_train, Y_train):
        reg = X_train.T @ X_train
        reg += np.eye(reg.shape[0]) * self.ridge_lambda
        self.B = la.inv(reg) @ X_train.T @ Y_train


    def predict(self, current_features):
        return (current_features @ self.B).flatten()


    def extract_causal_matrix(self):
        return np.abs(self.B[:self.num_signals,:])


class PluginNonlinearNVAR(PluginLinearVAR):

    def _expand(self,X):
        return np.hstack([X,np.tanh(X**2)])


    def fit(self,X_train,Y_train):
        X=self._expand(X_train)
        reg=X.T@X + np.eye(X.shape[1])*self.ridge_lambda
        self.B=la.inv(reg)@X.T@Y_train


    def predict(self,current_features):
        return (self._expand(current_features)@self.B).flatten()
