import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_parquet("energy_training_table.parquet")

# Clean comparison:
# 60 GeV hadrons, unlocked phase, no tungsten
plot_df = df[
    (df["particle"] == "h") &
    (df["energy_GeV"] == 60) &
    (df["phase_mode"] == "unlocked") &
    (df["tungsten"] == "no")
]

summary = plot_df.groupby("sipm_bias")["adc_sum"].agg(["mean", "std"])

print(summary)

# --- Get script name and create output folder --- #

script_full_path = os.path.realpath(__file__)
script_name = os.path.basename(script_full_path)

output_folder = f"dump/{os.path.splitext(script_name)[0]}"

os.makedirs(output_folder, exist_ok=True)

plt.figure(figsize=(6, 4))
plt.bar(summary.index.astype(str), summary["mean"], yerr=summary["std"], capsize=5)
plt.xlabel("SiPM bias [V]")
plt.ylabel("Mean ADC sum")
plt.title("60 GeV hadrons: ADC response vs SiPM bias")
plt.tight_layout()
plot_path = os.path.join(
    output_folder,
    "adc_sum_vs_sipm_bias_60GeV.png"
)

plt.savefig(plot_path, dpi=300)
print(f"Saved: {plot_path}")
plt.show()