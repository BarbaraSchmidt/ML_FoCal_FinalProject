import argparse
import os
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def make_output_folder():
    script_full_path = os.path.realpath(__file__)
    script_name = os.path.basename(script_full_path)
    output_folder = os.path.join("dump", os.path.splitext(script_name)[0])
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def split_within_each_run(df, test_size=0.10, random_state=42):
    train_idx = []
    test_idx = []

    for run, run_df in df.groupby("run"):
        run_indices = run_df.index.to_numpy()

        run_train_idx, run_test_idx = train_test_split(
            run_indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

        train_idx.extend(run_train_idx)
        test_idx.extend(run_test_idx)

    return np.array(train_idx), np.array(test_idx)


def make_pipeline(X):
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

    return Pipeline([
        ("preprocess", preprocess),
        ("model", model),
    ])


def get_X_y(df, target):
    drop_cols = ["run", "event_index", target, "particle"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target]
    return X, y


def train_and_evaluate(name, train_df, test_df, target):
    X_train, y_train = get_X_y(train_df, target)
    X_test, y_test = get_X_y(test_df, target)

    pipe = make_pipeline(X_train)
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    result = pd.DataFrame({
        "model": name,
        "run": test_df["run"].values,
        "true_energy_GeV": y_test.values,
        "pred_energy_GeV": pred,
        "error_GeV": pred - y_test.values,
        "abs_error_GeV": np.abs(pred - y_test.values),
    })

    metrics = {
        "model": name,
        "n_train_events": len(train_df),
        "n_test_events": len(test_df),
        "train_runs": ",".join(map(str, sorted(train_df["run"].unique()))),
        "test_runs": ",".join(map(str, sorted(test_df["run"].unique()))),
        "mae_GeV": mae,
        "rmse_GeV": rmse,
    }

    return metrics, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="energy_training_table.parquet")
    parser.add_argument("--target", default="energy_GeV")
    args = parser.parse_args()

    output_folder = make_output_folder()

    df = pd.read_parquet(args.input)
    df = df[df["particle"] == "h"].copy()

    epical_df = df[
        (df["phase_mode"] == "unlocked") &
        (df["tungsten"] == "no")
    ].copy()

    standalone_df = df[
        (df["phase_mode"] == "locked_5") &
        (df["tungsten"] == "yes")
    ].copy()

    combined_df = pd.concat([epical_df, standalone_df], ignore_index=False)

    standalone_train_idx, standalone_test_idx = split_within_each_run(
        standalone_df,
        test_size=0.10,
        random_state=42,
    )

    standalone_train_df = standalone_df.loc[standalone_train_idx].copy()
    standalone_test_df = standalone_df.loc[standalone_test_idx].copy()

    experiments = {
        "standalone_only": standalone_train_df,
        "epical_only": epical_df,
        "standalone_plus_epical": pd.concat(
            [standalone_train_df, epical_df],
            ignore_index=False,
        ),
    }

    all_metrics = []
    all_predictions = []

    for name, train_df in experiments.items():
        print(f"\nTraining: {name}")
        print(f"Train events: {len(train_df)}")
        print(f"Train runs: {sorted(train_df['run'].unique())}")
        print(f"Test events: {len(standalone_test_df)}")
        print(f"Test runs: {sorted(standalone_test_df['run'].unique())}")

        metrics, predictions = train_and_evaluate(
            name,
            train_df,
            standalone_test_df,
            args.target,
        )

        all_metrics.append(metrics)
        all_predictions.append(predictions)

        print(f"MAE:  {metrics['mae_GeV']:.3f} GeV")
        print(f"RMSE: {metrics['rmse_GeV']:.3f} GeV")

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    metrics_path = os.path.join(output_folder, "dataset_comparison_metrics.csv")
    predictions_path = os.path.join(output_folder, "dataset_comparison_predictions.csv")

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    print("\nSaved:")
    print(metrics_path)
    print(predictions_path)

    print("\nSummary:")
    print(metrics_df[["model", "mae_GeV", "rmse_GeV", "n_train_events", "n_test_events"]])

    print("\nPer-run mean prediction:")
    print(
        predictions_df
        .groupby(["model", "run"])[["true_energy_GeV", "pred_energy_GeV", "abs_error_GeV"]]
        .mean()
    )


if __name__ == "__main__":
    main()