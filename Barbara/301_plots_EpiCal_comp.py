import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# Output folder
# -----------------------------

script_full_path = os.path.realpath(__file__)
script_name = os.path.basename(script_full_path)

output_folder = os.path.join(
    "dump",
    os.path.splitext(script_name)[0]
)

os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# Read metrics
# -----------------------------

df = pd.read_csv(
    "dump/101_train_energy_regressor_Epical_test/dataset_comparison_metrics.csv"
)

print(df)




# -----------------------------
# MAE plot
# -----------------------------

plt.figure(figsize=(7,5))

plt.bar(
    df["model"],
    df["mae_GeV"]
)

plt.ylabel("MAE [GeV]")
plt.xlabel("Training dataset")
plt.title("Standalone test performance")

plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "mae_comparison.png"),
    dpi=300
)

plt.show()


plt.figure(figsize=(7,5))

plt.bar(
    df["model"],
    df["rmse_GeV"]
)

plt.ylabel("RMSE [GeV]")
plt.xlabel("Training dataset")
plt.title("Standalone test performance")

plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "rmse_comparison.png"),
    dpi=300
)

plt.show()



# predicted vs true energy


pred = pd.read_csv(
    "dump/101_train_energy_regressor_Epical_test/dataset_comparison_predictions.csv"
)



summary = (
    pred.groupby(["model", "run"])
    [["true_energy_GeV", "pred_energy_GeV"]]
    .mean()
    .reset_index()
)



plt.figure(figsize=(7,7))

for model in summary["model"].unique():

    sub = summary[summary["model"] == model]

    plt.scatter(
        sub["true_energy_GeV"],
        sub["pred_energy_GeV"],
        label=model
    )

plt.plot(
    [0, 400],
    [0, 400],
    "--"
)

plt.xlabel("True energy [GeV]")
plt.ylabel("Predicted energy [GeV]")
plt.title("Prediction performance")
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "predicted_vs_true.png"),
    dpi=300
)

plt.show()



plt.figure(figsize=(7,7))

for model in summary["model"].unique():

    sub = summary[summary["model"] == model]

    plt.scatter(
        sub["true_energy_GeV"],
        sub["pred_energy_GeV"],
        label=model
    )

plt.plot(
    [0, 400],
    [0, 400],
    "--"
)

plt.xlabel("True energy [GeV]")
plt.ylabel("Predicted energy [GeV]")
plt.title("Prediction performance")
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "predicted_vs_true.png"),
    dpi=300
)

plt.show()


summary = (
    pred.groupby(["model", "run"])
    .agg(
        true_energy_GeV=("true_energy_GeV", "mean"),
        pred_mean=("pred_energy_GeV", "mean"),
        pred_std=("pred_energy_GeV", "std"),
    )
    .reset_index()
)

plt.figure(figsize=(8,8))

for model in summary["model"].unique():

    sub = summary[summary["model"] == model]

    plt.errorbar(
        sub["true_energy_GeV"],
        sub["pred_mean"],
        yerr=sub["error_std"],
        fmt="o",
        capsize=4,
        label=model,
    )

plt.plot(
    [0, 400],
    [0, 400],
    "--",
    label="Perfect prediction"
)

plt.xlabel("True energy [GeV]")
plt.ylabel("Predicted energy [GeV]")
plt.title("Predicted vs True Energy")
plt.legend()
plt.tight_layout()


summary = (
    pred.groupby(["model", "run"])
    .agg(
        true_energy_GeV=("true_energy_GeV", "mean"),
        pred_mean=("pred_energy_GeV", "mean"),
        pred_std=("pred_energy_GeV", "std"),
    )
    .reset_index()
)


plt.figure(figsize=(8,8))

for model in summary["model"].unique():

    sub = summary[summary["model"] == model]

    plt.errorbar(
        sub["true_energy_GeV"],
        sub["pred_mean"],
        yerr=sub["error_std"],
        fmt="o",
        capsize=4,
        label=model,
    )

plt.plot(
    [0, 400],
    [0, 400],
    "--",
    label="Perfect prediction"
)

plt.xlabel("True energy [GeV]")
plt.ylabel("Predicted energy [GeV]")
plt.title("Predicted vs True Energy")
plt.legend()
plt.tight_layout()