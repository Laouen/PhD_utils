from sklearn.preprocessing import RobustScaler, StandardScaler
import numpy as np

class RobustClipper:

    def __init__(self, q: float) -> None:
        self.q = q
        self.scaler = RobustScaler()
    
    def fit(self, X: np.ndarray):
        X = self.scaler.fit_transform(X)
        self.q1s = [np.quantile(X[:,i], self.q) for i in range(X.shape[1])]
        self.q2s = [np.quantile(X[:,i], 1-self.q) for i in range(X.shape[1])]

    def transform(self, X: np.ndarray):

        # |samples| x |features|
        X = self.scaler.transform(X)

        n_features = X.shape[1]

        for i, q1, q2 in zip(range(n_features), self.q1s, self.q2s):
            x_ft = X[:,i]
            X[:,i] = np.clip(x_ft, q1, q2)
        
        X = self.scaler.inverse_transform(X)
        return X

    def fit_transform(self, X: np.ndarray):

        self.fit(X)
        return self.transform(X)

class ClipStandardScaler:

    def __init__(self, q=0.001):
        self.q = q
        self.robust_clipper = RobustClipper(q)
        self.standard_scaler = StandardScaler()
    
    def fit(self, X: np.ndarray):
        X = self.robust_clipper.fit_transform(X)
        self.standard_scaler.fit(X)

    def transform(self, X: np.ndarray):
        X = self.robust_clipper.transform(X)
        return self.standard_scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        return self.transform(X)