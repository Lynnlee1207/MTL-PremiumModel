import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .base import PremiumModel


class GBMPremiumModel(PremiumModel):
    cat_encode_mode = "dummies"
    fit_scaler = False  # scaler is already included in HistGradientBoostingRegressor

    def __init__(self, frequency_params={}, severity_params={}):
        self.frequency_model = HistGradientBoostingRegressor(loss="poisson", **frequency_params)
        self.severity_model = HistGradientBoostingRegressor(loss="gamma", **severity_params)  # type: ignore

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
