import numpy as np
import pandas as pd
import pyreadr
from sklearn.model_selection import StratifiedKFold

# Load the Belgian MTPL data
df: pd.DataFrame = pyreadr.read_r("data/mtpl_data.rds")[None].set_index("id")

# Define features
all_features = np.array(["ageph", "bm", "agec", "power", "long", "lat", "coverage", "fuel", "sex", "use", "fleet"])
categorical_mask = df[all_features].dtypes == "category"
categorical_features = all_features[categorical_mask]
continuous_features = all_features[~categorical_mask]
continuous_features_idx = np.arange(len(all_features))[~categorical_mask]

# Split folds for cross-validation
RANDOM_STATE = 123
n_folds = 6
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
df["fold"] = None
for i, (_, test_idx) in enumerate(skf.split(df, df["nclaims"]), start=1):
    df.loc[test_idx + 1, "fold"] = i


# Define feature sets for frequency and severity models
X_freq = df[["coverage", "fuel", "ageph", "power", "bm", "long", "lat"]]
y_freq = df["nclaims"] / df["expo"]
w_freq = df["expo"]

X_sev = df[["coverage", "ageph", "bm"]]
y_sev = df["average"]
w_sev = df["nclaims"]

sev_mask = df["nclaims"] > 0
