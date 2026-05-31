
import argparse
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="energy_training_table.parquet")
    parser.add_argument("--target", default="energy_GeV")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)

    # Do not use run or event index as model features.
    drop_cols = ["run", "event_index", args.target]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[args.target]
    groups = df["run"]

    categorical = []
    numeric = []

    for col in X.columns:
        if X[col].dtype == "object" or col in ["tungsten"]:
            categorical.append(col)
        else:
            numeric.append(col)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ]
    )

    model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.03,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=42,
    )

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", model),
    ])

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = pipe.predict(X.iloc[test_idx])

    mae = mean_absolute_error(y.iloc[test_idx], pred)
    rmse = mean_squared_error(y.iloc[test_idx], pred, squared=False)

    print("Run-wise split evaluation")
    print(f"Train runs: {sorted(df.iloc[train_idx]['run'].unique())}")
    print(f"Test runs:  {sorted(df.iloc[test_idx]['run'].unique())}")
    print(f"MAE:  {mae:.3f} GeV")
    print(f"RMSE: {rmse:.3f} GeV")

    result = pd.DataFrame({
        "run": df.iloc[test_idx]["run"].values,
        "true_energy_GeV": y.iloc[test_idx].values,
        "pred_energy_GeV": pred,
    })
    print(result.groupby("run")[["true_energy_GeV", "pred_energy_GeV"]].mean())


if __name__ == "__main__":
    main()
