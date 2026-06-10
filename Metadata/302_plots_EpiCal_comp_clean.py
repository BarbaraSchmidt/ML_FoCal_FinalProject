import os
import pandas as pd
import matplotlib.pyplot as plt


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
# Input files
# -----------------------------

input_folder = "dump/101_train_energy_regressor_Epical_test"

metrics_path = os.path.join(input_folder, "dataset_comparison_metrics.csv")
predictions_path = os.path.join(input_folder, "dataset_comparison_predictions.csv")

metrics = pd.read_csv(metrics_path)
pred = pd.read_csv(predictions_path)

print(metrics)


# -----------------------------
# Nice model labels
# -----------------------------

label_map = {
    "standalone_only": "Standalone only",
    "epical_only": "Epical only",
    "standalone_plus_epical": "Standalone + Epical",
}

metrics["model_label"] = metrics["model"].map(label_map)
pred["model_label"] = pred["model"].map(label_map)

model_order = [
    "Standalone only",
    "Standalone + Epical",
    "Epical only",
]


# -----------------------------
# MAE plot
# -----------------------------

metrics_sorted = metrics.set_index("model_label").loc[model_order].reset_index()

plt.figure(figsize=(7, 5))

bars = plt.bar(
    metrics_sorted["model_label"],
    metrics_sorted["mae_GeV"]
)

for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.1f}",
        ha="center",
        va="bottom"
    )

plt.ylabel("MAE [GeV]")
plt.xlabel("Training dataset")
plt.title("Standalone test performance: MAE")
plt.xticks(rotation=20, ha="right")

plt.ylim(0, 1.15 * metrics_sorted["mae_GeV"].max())

plt.tight_layout()

plot_path = os.path.join(output_folder, "mae_comparison.png")
plt.savefig(plot_path, dpi=300)
print(f"Saved: {plot_path}")
plt.show()


# -----------------------------
# RMSE plot
# -----------------------------

plt.figure(figsize=(7, 5))

bars = plt.bar(
    metrics_sorted["model_label"],
    metrics_sorted["rmse_GeV"]
)

for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.1f}",
        ha="center",
        va="bottom"
    )

plt.ylabel("RMSE [GeV]")
plt.xlabel("Training dataset")
plt.title("Standalone test performance: RMSE")
plt.xticks(rotation=20, ha="right")

plt.ylim(0, 1.15 * metrics_sorted["rmse_GeV"].max())

plt.tight_layout()

plot_path = os.path.join(output_folder, "rmse_comparison.png")
plt.savefig(plot_path, dpi=300)
print(f"Saved: {plot_path}")
plt.show()


# -----------------------------
# Predicted vs true, averaged by run
# Error bars show event-by-event prediction spread
# -----------------------------

summary = (
    pred.groupby(["model_label", "true_energy_GeV"])
    .agg(
        pred_mean=("pred_energy_GeV", "mean"),
        pred_std=("pred_energy_GeV", "std"),
        n_events=("pred_energy_GeV", "size"),
    )
    .reset_index()
)

plt.figure(figsize=(8, 8))

marker_list = ["o", "v", "^"]

for model_label in model_order:

    marker = marker_list[model_order.index(model_label)]

    sub = summary[summary["model_label"] == model_label]

    plt.errorbar(
        sub["true_energy_GeV"],
        sub["pred_mean"],
        yerr=sub["pred_std"],
        marker=marker,
        linestyle="none",
        capsize=4,
        label=model_label,
    )
plt.plot(
    [0, 400],
    [0, 400],
    "--",
    label="Perfect prediction"
)

plt.xlabel("True energy [GeV]")
plt.ylabel("Mean predicted energy [GeV]")
plt.title("Predicted vs true energy on standalone test set")
plt.legend()
plt.tight_layout()

plot_path = os.path.join(output_folder, "predicted_vs_true_with_errorbars.png")
plt.savefig(plot_path, dpi=300)
print(f"Saved: {plot_path}")
plt.show()


# -----------------------------
# Optional cleaner version:
# Error bars show uncertainty on the mean, not full spread
# -----------------------------

summary["pred_sem"] = summary["pred_std"] / summary["n_events"] ** 0.5

plt.figure(figsize=(8, 8))

for model_label in model_order:
    sub = summary[summary["model_label"] == model_label]

    plt.errorbar(
        sub["true_energy_GeV"],
        sub["pred_mean"],
        yerr=sub["pred_sem"],
        fmt="o",
        capsize=4,
        label=model_label,
    )

plt.plot(
    [0, 400],
    [0, 400],
    "--",
    label="Perfect prediction"
)

plt.xlabel("True energy [GeV]")
plt.ylabel("Mean predicted energy [GeV]")
plt.title("Predicted vs true energy on standalone test set")
plt.legend()
plt.tight_layout()

plot_path = os.path.join(output_folder, "predicted_vs_true_mean_error.png")
plt.savefig(plot_path, dpi=300)
print(f"Saved: {plot_path}")
plt.show()