"""Vẽ lại Hình 2, 3, 4 của báo cáo (Hình 1 là sơ đồ pipeline vẽ bằng TikZ trong LaTeX, không sinh từ dữ liệu).

    python scripts/make_figures.py --pred-dir outputs/checkpoint_eval

Đầu ra: outputs/figures/fig2_ac_vs_irradiation.pdf, fig3_pred_vs_actual_plant2.pdf,
fig4_lstm_residuals_plant2.pdf (+ bản .png)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import load_config, p  # noqa: E402
from src.evaluate import load_preds, to_plant  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})
COL = {"LSTM": "#4a3aa7", "ARIMA": "#eda100", "RandomForest": "#eb6834", "measured": "black"}


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ->", out / f"{name}.pdf")


def fig2(cfg, out):
    f = pd.read_csv(p(cfg["paths"]["features_csv"]))
    thr = cfg["data"]["anomaly_irradiation"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    norm = LogNorm(vmin=1, vmax=1e4)
    for ax, (pl, g) in zip(axes, f.groupby("PLANT_ID")):
        hb = ax.hexbin(g.IRRADIATION, g.AC_POWER, gridsize=45, cmap="Blues", norm=norm, mincnt=1,
                       extent=(0, 1.25, 0, 1450), linewidths=0)
        an = g[(g.IRRADIATION > thr) & (g.AC_POWER == 0)]
        ax.scatter(an.IRRADIATION, an.AC_POWER, s=4, c="#d62728", lw=0, zorder=3)
        name = "Plant 1" if pl == cfg["data"]["dc_fix_plant"] else "Plant 2"
        ax.set_title(f"{name} ({pl}), n = {len(g):,}")
        ax.text(0.03, 0.95, f"Anomalous: {len(an):,}", transform=ax.transAxes, va="top", color="#d62728")
        ax.set_xlabel("Irradiation")
    axes[0].set_ylabel("AC power (kW)")
    fig.colorbar(hb, ax=axes, label="Records per bin (log scale)", pad=0.02)
    save(fig, out, "fig2_ac_vs_irradiation")


def fig3(plant, out):
    g = plant[plant.PLANT_ID_4136001 == 1].sort_values("DATE_TIME")
    day = pd.Timestamp("2020-06-14")
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.4))
    for ax, d, title in ((axes[0], g, "(a) Plant 2 (4136001), full test period"),
                         (axes[1], g[(g.DATE_TIME >= day) & (g.DATE_TIME < day + pd.Timedelta(days=1))],
                          "(b) Plant 2, 14 June (shaded in a)")):
        for m, lab in (("LSTM", "LSTM"), ("ARIMA", "ARIMAX"), ("RandomForest", "Random Forest")):
            ax.plot(d.DATE_TIME, d[m], color=COL[m], lw=0.9, label=lab)
        ax.plot(d.DATE_TIME, d.y_true, color="black", lw=1.1, label="Measured")
        ax.set_title(title, loc="left")
        ax.set_ylabel("Mean AC power (kW)")
        ax.set_ylim(0, 1300)
    axes[0].axvspan(day, day + pd.Timedelta(days=1), color="grey", alpha=0.15)
    axes[0].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    axes[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    axes[1].set_xlabel("Time of day")
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.32), frameon=False)
    fig.tight_layout()
    save(fig, out, "fig3_pred_vs_actual_plant2")


def fig4(cfg, inv, out):
    ids = pd.read_csv(p(cfg["paths"]["inverter_ids"]))
    le = LabelEncoder().fit(ids.SOURCE_KEY)
    ids["SOURCE_KEY_ENCODED"] = le.transform(ids.SOURCE_KEY)
    d = inv[(inv.PLANT_ID_4136001 == 1) & (inv.IRR_RAW > 0)].merge(ids, on="SOURCE_KEY_ENCODED")
    d["resid"] = d.AC_POWER - d.LSTM
    order = sorted(d.inverter_id.unique())                                 # P2-01 ... P2-22
    data = [d.loc[d.inverter_id == k, "resid"].values for k in order]
    fig, ax = plt.subplots(figsize=(4.2, 4.6))
    bp = ax.boxplot(data[::-1], vert=False, widths=0.55, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=3),
                    medianprops=dict(color="black", lw=1.2), flierprops=dict(marker=".", markersize=2, alpha=0.4))
    for box, k in zip(bp["boxes"], order[::-1]):
        box.set_facecolor("#f28e2b" if k == "P2-01" else "#c6dbef")
    ax.set_yticks(range(1, len(order) + 1), order[::-1])
    ax.axvline(0, color="grey", lw=0.6, ls="--")
    ax.set_xlabel("Residual: measured − LSTM prediction (kW), daytime records")
    save(fig, out, "fig4_lstm_residuals_plant2")
    s = d.groupby("inverter_id").resid
    summary = pd.DataFrame({"mean": s.mean(), "median": s.median(), "q1": s.quantile(0.25),
                            "share_below_-300": s.apply(lambda r: (r < -300).mean())}).round(3)
    summary.to_csv(out / "fig4_residual_summary.csv")
    return summary


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default=f"{cfg['paths']['outputs']}/checkpoint_eval")
    ap.add_argument("--out", default=f"{cfg['paths']['outputs']}/figures")
    a = ap.parse_args()
    out = p(a.out); out.mkdir(parents=True, exist_ok=True)
    inv, ts = load_preds(p(a.pred_dir))
    plant = to_plant(inv, ts)
    fig2(cfg, out)
    fig3(plant, out)
    s = fig4(cfg, inv, out)
    print("\nPhần dư LSTM ban ngày, Plant 2 (Mục 7.6):")
    print(s.loc[["P2-01"]].to_string(), f"\nTrung vị theo inverter: {s['median'].min():.1f} .. {s['median'].max():.1f} kW")


if __name__ == "__main__":
    main()
