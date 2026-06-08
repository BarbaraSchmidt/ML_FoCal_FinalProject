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

output_folder = os.path.join(
    "dump",
    os.path.splitext(script_name)[0]
)

os.makedirs(output_folder, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="energy_training_table.parquet")
    parser.add_argument("--target", default="energy_GeV")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)

    # Hadron energy regression only
    df = df[df["particle"] == "h"].copy()

    # Use only unlocked, no-tungsten / no-Epical-style data
    df = df[
        (df["phase_mode"] == "unlocked") &
        (df["tungsten"] == "no")
    ].copy()

    print(f"Using {len(df)} events")
    print(f"Runs: {[int(r) for r in sorted(df['run'].unique())]}")
    print(df.groupby("energy_GeV")["run"].nunique())

    drop_cols = ["run", "event_index", args.target, "particle"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[args.target]

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

    print("90/10 split within each run")
    print(f"Train runs: {[int(r) for r in sorted(df.loc[train_idx]['run'].unique())]}")
    print(f"Test runs:  {[int(r) for r in sorted(df.loc[test_idx]['run'].unique())]}")
    print(f"MAE:  {mae:.3f} GeV")
    print(f"RMSE: {rmse:.3f} GeV")

    result = pd.DataFrame({
        "run": df.loc[test_idx]["run"].values,
        "true_energy_GeV": y.loc[test_idx].values,
        "pred_energy_GeV": pred,
    })

    # -----------------------------
    # Predicted vs true energy
    # -----------------------------

    energy_summary = (
        result.groupby("true_energy_GeV")["pred_energy_GeV"]
        .agg(["mean", "std"])
        .reset_index()
    )

    coeff = np.polyfit(
        energy_summary["true_energy_GeV"],
        energy_summary["mean"],
        1
    )

    slope = coeff[0]
    intercept = coeff[1]

    print(f"Slope     = {slope:.4f}")
    print(f"Intercept = {intercept:.4f} GeV")

    plt.figure(figsize=(6, 6))

    plt.errorbar(
        energy_summary["true_energy_GeV"],
        energy_summary["mean"],
        yerr=energy_summary["std"],
        fmt="o",
        capsize=4,
        label="Mean prediction"
    )

    x = np.linspace(50, 320, 100)

    plt.plot(
        x,
        slope * x + intercept,
        "-",
        label=f"Fit: y={slope:.3f}x+{intercept:.1f}"
    )

    plt.plot(
        [0, 350],
        [0, 350],
        "--",
        label="Perfect prediction"
    )

    plt.xlabel("True energy [GeV]")
    plt.ylabel("Predicted energy [GeV]")
    plt.title("Predicted vs true energy")
    plt.legend()

    plt.tight_layout()

    plot_path = os.path.join(output_folder, "predicted_vs_true.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved: {plot_path}")
    plt.show()

    print(result.groupby("run")[["true_energy_GeV", "pred_energy_GeV"]].mean())

    # -----------------------------
    # Energy resolution plot
    # -----------------------------

    energy_summary["resolution"] = (
        energy_summary["std"] / energy_summary["true_energy_GeV"]
    )

    plt.figure(figsize=(6, 4))

    plt.plot(
        energy_summary["true_energy_GeV"],
        energy_summary["resolution"],
        "o-"
    )

    plt.xlabel("True energy [GeV]")
    plt.ylabel(r"$\sigma(E_{\mathrm{pred}}) / E_{\mathrm{true}}$")
    plt.title("Relative energy resolution")

    plt.tight_layout()

    plot_path = os.path.join(output_folder, "energy_resolution.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved: {plot_path}")
    plt.show()

    print(energy_summary[["true_energy_GeV", "mean", "std", "resolution"]])

    # -----------------------------
    # Feature correlation with energy
    # -----------------------------

    corr_features = [
        "adc_sum",
        "n_adc_hit_100",
        "tot_sum",
        "adc_max",
        "tot_max",
        "n_tot_hit",
        "channel_width_adc",
        "channel_cog_adc",
    ]

    corr_features = [c for c in corr_features if c in df.columns]

    corr = (
        df[corr_features + [args.target]]
        .corr(numeric_only=True)[args.target]
        .drop(args.target)
        .sort_values()
    )

    plt.figure(figsize=(7, 5))

    plt.barh(
        corr.index,
        corr.values
    )

    plt.xlabel("Correlation with beam energy")
    plt.ylabel("Feature")
    plt.title("Feature correlation with energy")

    plt.tight_layout()

    plot_path = os.path.join(output_folder, "feature_correlation.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved: {plot_path}")
    plt.show()

    print(corr.sort_values(ascending=False))

    # -----------------------------
    # ADC response vs energy
    # -----------------------------

    adc_summary = (
        df.groupby("energy_GeV")["adc_sum"]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(6, 4))

    plt.errorbar(
        adc_summary["energy_GeV"],
        adc_summary["mean"],
        yerr=adc_summary["std"],
        fmt="o",
        capsize=4,
    )

    plt.xlabel("Beam energy [GeV]")
    plt.ylabel("ADC sum")
    plt.title("ADC response vs beam energy")

    plt.tight_layout()

    plot_path = os.path.join(output_folder, "adc_sum_vs_energy.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved: {plot_path}")
    plt.show()

    print(adc_summary)


if __name__ == "__main__":
    main()