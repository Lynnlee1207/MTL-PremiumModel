import joblib
import pandas as pd
from sklearn.linear_model import GammaRegressor, PoissonRegressor

from .base import PremiumModel


class GLMPremiumModel(PremiumModel):
    cat_encode_mode = "dummies"
    fit_scaler = False

    def __init__(self, max_iter=1000, alpha=0.0):
        self.frequency_model = PoissonRegressor(max_iter=max_iter, alpha=alpha, fit_intercept=False)
        self.severity_model = GammaRegressor(max_iter=max_iter, alpha=alpha, fit_intercept=True)

    def _fit(self, X_freq, y_freq, w_freq, X_sev, y_sev, w_sev):
        self.frequency_model.fit(X_freq.to_numpy(), y_freq.to_numpy(), sample_weight=w_freq.to_numpy())
        self.severity_model.fit(X_sev.to_numpy(), y_sev.to_numpy(), sample_weight=w_sev.to_numpy())

    def _predict(self, X_freq, X_sev):
        freq_pred = self.frequency_model.predict(X_freq.to_numpy())
        sev_pred = self.severity_model.predict(X_sev.to_numpy())
        return pd.Series(freq_pred, index=X_freq.index, name="frequency"), pd.Series(sev_pred, index=X_sev.index, name="severity")

    def _save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def _load(cls, path: str):
        return joblib.load(path)
