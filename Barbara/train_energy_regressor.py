
import argparse
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="energy_training_table.parquet")
    parser.add_argument("--target", default="energy_GeV")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    df = df[df["particle"] == "h"].copy()

    drop_cols = ["run", "event_index", args.target, "particle"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[args.target]
    groups = df["run"]

    categorical = ["phase_mode", "tungsten"]
    categorical = [c for c in categorical if c in X.columns]

    numeric = [c for c in X.columns if c not in categorical]

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

    train_idx = []
    test_idx = []

    for run, run_df in df.groupby("run"):
        run_indices = run_df.index.to_numpy()

        run_train_idx, run_test_idx = train_test_split(
            run_indices,
            test_size=0.10,
            random_state=42,
            shuffle=True,
        )

        train_idx.extend(run_train_idx)
        test_idx.extend(run_test_idx)

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    pipe.fit(X.loc[train_idx], y.loc[train_idx])
    # feature_names = pipe.named_steps["preprocess"].get_feature_names_out()

    # importance = pipe.named_steps["model"].feature_importances_

    # imp = pd.DataFrame({
    #     "feature": feature_names,
    #     "importance": importance
    # }).sort_values("importance", ascending=False)

    # print(imp.head(20))


    pred = pipe.predict(X.loc[test_idx])

    mae = mean_absolute_error(y.loc[test_idx], pred)
    rmse = np.sqrt(mean_squared_error(y.loc[test_idx], pred))

    print("90/10 split within each run")
    print(f"Train runs: {sorted(df.loc[train_idx]['run'].unique())}")
    print(f"Test runs:  {sorted(df.loc[test_idx]['run'].unique())}")
    print(f"MAE:  {mae:.3f} GeV")
    print(f"RMSE: {rmse:.3f} GeV")

    result = pd.DataFrame({
        "run": df.loc[test_idx]["run"].values,
        "true_energy_GeV": y.loc[test_idx].values,
        "pred_energy_GeV": pred,
    })
    print(result.groupby("run")[["true_energy_GeV", "pred_energy_GeV"]].mean())


if __name__ == "__main__":
    main()
