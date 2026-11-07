"""Figure of the tuned 20 dB return-loss nominal (run tune20_4): Palace vs the analytic prior.

    python tools_plot_rl20.py   -> figs/rl20_tuned.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = Path(__file__).resolve().parent          # experiments/: this script, its inputs and its logs
HERE = EXP.parent                              # the family folder: runs/, data/, figs/, samples.json
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP))
import prior  # noqa: E402

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#e4e3dd"

z = np.load(HERE / "data" / "aux" / "tune20_4.npz")
f, s11, s21, p = z["f"], z["S11"], z["S21"], z["p"]
q11, q21 = prior.s_prior(p, f)
db = lambda s: 20 * np.log10(np.maximum(np.abs(s), 1e-12))

fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True, constrained_layout=True)
for ax in axes:
    ax.axvspan(4.875, 5.125, color="#eeede8", lw=0)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)

ax = axes[0]
ax.plot(f, db(s21), color=BLUE, lw=2, label="Palace (FEM)")
ax.plot(f, db(q21), color=ORANGE, lw=2, ls="--", label="analytic prior")
ax.set_ylim(-40, 2); ax.set_ylabel("|S21| (dB)", color=INK)
ax.legend(frameon=False, loc="upper left", labelcolor=INK)
ax.set_title("Edge-coupled BPF, N = 3, 20 dB return-loss design (tuned in Palace)", color=INK, fontsize=11, loc="left")
k = np.argmax(db(s21))
ax.annotate(f"IL {-db(s21)[k]:.2f} dB", (f[k], db(s21)[k]), xytext=(f[k] + 0.25, -6), color=INK, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

ax = axes[1]
ax.plot(f, db(s11), color=BLUE, lw=2)
ax.plot(f, db(q11), color=ORANGE, lw=2, ls="--")
ax.axhline(-20, color=INK, lw=1, ls=":")
ax.text(f[0] + 0.02, -19, "-20 dB target", color=INK, fontsize=9)
ax.set_ylim(-45, 2); ax.set_ylabel("|S11| (dB)", color=INK); ax.set_xlabel("frequency (GHz)", color=INK)
ax.text(5.0, -43, "design band 4.875-5.125 GHz", color=MUTED, fontsize=8, ha="center")
w1, w2, g1, g2, l1, l2 = p[:6]
ax.text(0.99, 0.04, f"w {w1:.3f}/{w2:.3f}  gap {g1:.3f}/{g2:.3f}  l {l1:.2f}/{l2:.2f} mm\nRO4350B-like, h 0.508 mm",
        transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=8)

out = HERE / "figs" / "rl20_tuned.png"
fig.savefig(out, dpi=160)
print(out)
