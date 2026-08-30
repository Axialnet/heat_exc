from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("//wsl.localhost/Ubuntu/home/venkat/projects/heat_exc")
V1_PATH = ROOT / "Data/version 1/shell_tube_fouling_final.csv"
V2_PATH = ROOT / "Data/version 2/v2_physics_corrected.csv"
OUT_PATH = ROOT / "Data/version 2/v1_vs_v2_distribution_plot.png"

COLUMNS = [
    "Re",
    "u_m_s",
    "tau_w_Pa",
]

v1 = pd.read_csv(V1_PATH)
v2 = pd.read_csv(V2_PATH)

fig, axes = plt.subplots(len(COLUMNS), 2, figsize=(14, 4.5 * len(COLUMNS)))
fig.suptitle("V1 vs V2 Distribution Comparison", fontsize=16, fontweight="bold")

for i, col in enumerate(COLUMNS):
    for j, (name, df) in enumerate([("V1", v1), ("V2", v2)]):
        ax = axes[i, j]
        if col not in df.columns:
            ax.text(0.5, 0.5, "Column missing", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        vals = df[col].dropna()
        ax.hist(vals, bins=45, alpha=0.8, edgecolor="black", color="steelblue" if name == "V1" else "darkorange")
        ax.set_title(f"{name} - {col}")
        ax.grid(alpha=0.25)
        if j == 0:
            ax.set_ylabel("Count")

        mean_val = vals.mean()
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.8, alpha=0.95)

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
print(f"Saved plot to: {OUT_PATH}")
