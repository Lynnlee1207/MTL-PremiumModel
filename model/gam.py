import joblib
import pandas as pd
from pygam import GammaGAM, PoissonGAM
from pygam.terms import TermList

from .base import PremiumModel


class GAMPremiumModel(PremiumModel):
    cat_encode_mode = "code"  # pygam supports categorical encoding via integer codes
    fit_scaler = False  # scaler is already included in pygam

    def __init__(self, frequency_terms, severity_terms):
        self.frequency_terms = frequency_terms
        self.severity_terms = severity_terms
        
    def _map_terms(self, terms, dtypes: pd.Series):
        from pygam import f, s, te
        cols = dtypes.index.tolist()
        idx_terms = TermList()
        for t in terms:
            if isinstance(t, tuple):
                idx_terms += te(*(cols.index(f) for f in t))
            elif dtypes[t].name == "category":
                idx_terms += f(cols.index(t))
            else:
                idx_terms += s(cols.index(t))
        return idx_terms

    def _fit(self, X_freq, y_freq, w_freq, X_sev, y_sev, w_sev):
        frequency_terms = self._map_terms(self.frequency_terms, X_freq.dtypes)
        severity_terms = self._map_terms(self.severity_terms, X_sev.dtypes)
        self.frequency_model = PoissonGAM(frequency_terms) # type: ignore
        self.severity_model = GammaGAM(severity_terms) # type: ignore
        self.frequency_model.fit(X_freq.to_numpy(), y_freq.to_numpy(), weights=w_freq.to_numpy())
        self.severity_model.fit(X_sev.to_numpy(), y_sev.to_numpy(), weights=w_sev.to_numpy())

    def _predict(self, X_freq, X_sev):
        freq_pred = self.frequency_model.predict(X_freq.to_numpy())
        sev_pred = self.severity_model.predict(X_sev.to_numpy())
        return pd.Series(freq_pred, index=X_freq.index, name="frequency"), pd.Series(sev_pred, index=X_sev.index, name="severity")

    def _save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def _load(cls, path: str):
        return joblib.load(path)
