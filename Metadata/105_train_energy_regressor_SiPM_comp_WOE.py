import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


# --- Get script name and create output folder --- #

script_full_path = os.path.realpath(__file__)
script_name = os.path.basename(script_full_path)
script_dir = os.path.dirname(script_full_path)

output_folder = os.path.join(
    script_dir,
    "dump",
    os.path.splitext(script_name)[0]
)

os.makedirs(output_folder, exist_ok=True)


def train_and_evaluate(df, label, target="energy_GeV"):
    drop_cols = ["run", "event_index", target, "particle"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target]

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
    pred = pipe.predict(X.loc[test_idx])

    mae = mean_absolute_error(y.loc[test_idx], pred)
    rmse = np.sqrt(mean_squared_error(y.loc[test_idx], pred))

    print()
    print(f"Training set: {label}")
    print(f"Events: {len(df)}")
    print(f"Runs: {[int(r) for r in sorted(df['run'].unique())]}")
    print(df.groupby("energy_GeV")["run"].nunique())
    print(f"MAE:  {mae:.3f} GeV")
    print(f"RMSE: {rmse:.3f} GeV")

    return {
        "training_set": label,
        "mae_GeV": mae,
        "rmse_GeV": rmse,
        "n_events": len(df),
        "n_runs": df["run"].nunique(),
        "energies": ", ".join(str(int(e)) for e in sorted(df["energy_GeV"].unique())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="energy_training_table.parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)

    # Standalone-style selected data only
    df = df[
        (df["particle"] == "h") &
        (df["phase_mode"] == "unlocked") &
        (df["tungsten"] == "no")
    ].copy()

    print(f"Using {len(df)} standalone events")
    print(f"Available SiPM biases: {sorted(df['sipm_bias'].unique())}")

    configs = [
        ("SiPM 43 V only", df[df["sipm_bias"] == 43].copy()),
        ("SiPM 45 V only", df[df["sipm_bias"] == 45].copy()),
        ("SiPM 43+45 V", df[df["sipm_bias"].isin([43, 45])].copy()),
    ]

    results = []

    for label, sub_df in configs:
        if len(sub_df) == 0:
            print(f"Skipping {label}: no events")
            continue

        results.append(train_and_evaluate(sub_df, label))

    results_df = pd.DataFrame(results)

    csv_path = os.path.join(output_folder, "sipm_bias_training_comparison.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    print(results_df)

    # -----------------------------
    # MAE comparison plot
    # -----------------------------

    plt.figure(figsize=(7, 5))

    bars = plt.bar(
        results_df["training_set"],
        results_df["mae_GeV"],
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
        )

    plt.ylabel("MAE [GeV]")
    plt.xlabel("Training dataset")
    plt.title("Energy regression performance vs SiPM bias selection")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1.15 * results_df["mae_GeV"].max())

    plt.tight_layout()

    plot_path = os.path.join(output_folder, "mae_vs_sipm_bias_selection.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved: {plot_path}")
    plt.show()

    # -----------------------------
    # Optional RMSE comparison plot
    # -----------------------------

    plt.figure(figsize=(7, 5))

    bars = plt.bar(
        results_df["training_set"],
        results_df["rmse_GeV"],
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
        )

    plt.ylabel("RMSE [GeV]")
    plt.xlabel("Training dataset")
    plt.title("Energy regression performance vs SiPM bias selection")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1.15 * results_df["rmse_GeV"].max())

    plt.tight_layout()

    plot_path = os.path.join(output_folder, "rmse_vs_sipm_bias_selection.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved: {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()