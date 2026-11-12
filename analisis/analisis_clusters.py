"""
Territorial digital-finance adoption in Mexico, 2018Q1-2026Q1
=============================================================
Replication and extension of Guzmán Ayala (2026, IPN-ESE thesis):
K-means typology of state-level financial access infrastructure (CNBV BDIF),
read through Rogers' (2003) Diffusion of Innovations, plus:
  * DBSCAN as a density-based robustness check / outlier detector
  * Territorial inequality (population-weighted Theil, between/within regions)
  * Stage-transition (Markov) matrices: pre-COVID, COVID, post-COVID
  * Exploratory event study: Banco del Bienestar expansion (SBD) as a
    place-based policy shock

Usage:  python3 analisis/analisis_clusters.py
Outputs: analisis/figuras/*.png (300 dpi) and analisis/tablas/*.csv
"""
from pathlib import Path
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score)
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "versiones_unidas_definitivas.csv"
FIG = ROOT / "analisis" / "figuras"
TAB = ROOT / "analisis" / "tablas"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

SEED = 42
K = 7

# ----------------------------------------------------------------------------
# Visual system (validated blue ramp; see README)
# ----------------------------------------------------------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#ffffff"
BLUE = "#2a78d6"          # single-series blue (slot 1)
BLUE_DARK = "#104281"
GRAY_MID = "#f0efec"
RED = "#e34948"           # diverging pole only
# Ordinal blue ramp, steps 100..700 (adjacent dE >= 9.1 incl. protan/deutan)
RAMP7 = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
              "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div", ["#0d366b", "#3987e5", "#cde2fb", GRAY_MID, "#f8c9c8", RED, "#8f1f1f"])

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 8,
    "axes.labelsize": 9, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "legend.fontsize": 8,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "savefig.dpi": 300, "savefig.bbox": "tight",
    "lines.linewidth": 2, "text.color": INK,
    "pdf.fonttype": 42, "ps.fonttype": 42,          # embed TrueType fonts (publisher-friendly)
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png")
    (FIG / "pdf").mkdir(exist_ok=True)
    fig.savefig(FIG / "pdf" / f"{name}.pdf")      # vector version for the publisher
    (FIG / "tiff").mkdir(exist_ok=True)
    fig.savefig(FIG / "tiff" / f"{name}.tif", dpi=600,  # Springer: mixed line art >= 600 dpi
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"  · figuras/{name}.png")


def note(fig, text, y=-0.02):
    width = int(fig.get_size_inches()[0] * 21)
    fig.text(0.01, y, textwrap.fill(text, width), ha="left", va="top", fontsize=7,
             color=MUTED, linespacing=1.4)


SOURCE = "Source: own elaboration with CNBV Financial Inclusion Databases (BDIF), 2018Q1–2026Q1."

# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------
VARS = ["SBC_x10K", "SBD_x10K", "SSoc_x10K", "SSof_x10K", "Corr_x10K",
        "CA_x10K", "TPV_x10K", "ETPV_x10K", "CBM_x10K"]
SHORT = {v: v.replace("_x10K", "") for v in VARS}
LONG = {
    "SBC_x10K": "Commercial bank branches",
    "SBD_x10K": "Development bank branches",
    "SSoc_x10K": "Socap branches",
    "SSof_x10K": "Sofipo branches",
    "Corr_x10K": "Banking agents",
    "CA_x10K": "ATMs",
    "TPV_x10K": "POS terminals",
    "ETPV_x10K": "Merchants with POS",
    "CBM_x10K": "Mobile-banking contracts",
}
DIGITAL = ["Corr_x10K", "CA_x10K", "TPV_x10K", "ETPV_x10K", "CBM_x10K"]
PHYSICAL = ["SBC_x10K", "SBD_x10K", "SSoc_x10K", "SSof_x10K"]

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["Fecha"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
df["q"] = df["date"].dt.to_period("Q")
df["qlab"] = df["q"].astype(str)
df["Region"] = df["Región"].replace({
    "Occidente y Bajío": "West & Bajío", "Noroeste": "Northwest", "Noreste": "Northeast",
    "Sur": "South", "Centro Sur y Oriente": "Centre-South & East",
    "Ciudad de México": "Mexico City"})
df = df.sort_values(["q", "Estado"]).reset_index(drop=True)
QUARTERS = sorted(df["q"].unique())
print(f"Panel: {df.Estado.nunique()} states × {len(QUARTERS)} quarters = {len(df)} obs.")


def period(q):
    if q <= pd.Period("2019Q4"):
        return "Pre-COVID (2018–2019)"
    if q <= pd.Period("2021Q4"):
        return "COVID (2020–2021)"
    return "Post-COVID (2022–2026)"


df["period"] = df["q"].apply(period)
PERIODS = ["Pre-COVID (2018–2019)", "COVID (2020–2021)", "Post-COVID (2022–2026)"]

desc = df[VARS].describe().T.rename(index=LONG)
desc.round(3).to_csv(TAB / "t1_descriptivos.csv")

# ----------------------------------------------------------------------------
# 2. Descriptive figures: national evolution & inequality
# ----------------------------------------------------------------------------
print("Descriptive figures…")

# Fig 1 — median and interquartile band across states, small multiples
fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.2), sharex=True)
for ax, v in zip(axes.flat, VARS):
    g = df.groupby("date")[v]
    med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
    lo, hi = g.min(), g.max()
    ax.fill_between(med.index, lo, hi, color="#eef4fc", lw=0)
    ax.fill_between(med.index, q1, q3, color="#b7d3f6", lw=0)
    ax.plot(med.index, med, color=BLUE_DARK, lw=1.8)
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"),
               color="#f6f6f4", zorder=0, lw=0)
    ax.axvline(pd.Timestamp("2022-03-31"), color=MUTED, lw=0.7, ls="-", alpha=0.6)
    ax.set_title(LONG[v], fontsize=8.5)
    ax.tick_params(axis="x", labelsize=7, rotation=0)
    ax.tick_params(axis="y", labelsize=7)
    ax.xaxis.set_major_locator(mpl.dates.YearLocator(2))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
handles = [Patch(color="#eef4fc", label="Min–max across states"),
           Patch(color="#b7d3f6", label="Interquartile range"),
           plt.Line2D([], [], color=BLUE_DARK, lw=1.8, label="Median state"),
           Patch(color="#f6f6f4", label="COVID-19 (2020–2021)")]
fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.035))
fig.suptitle("Financial access points per 10,000 adults across Mexican states",
             x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout(rect=(0, 0.02, 1, 0.97))
note(fig, SOURCE + " Vertical line: BDIF version 3 (2022Q1), methodological change in POS series.",
     y=-0.05)
save(fig, "fig01_evolucion_canales")


# Population-weighted Theil T and between/within-region decomposition
def theil(y, w):
    y, w = np.asarray(y, float), np.asarray(w, float)
    w = w / w.sum()
    mu = (w * y).sum()
    r = y / mu
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(r > 0, w * r * np.log(r), 0.0)
    return t.sum()


rows = []
for q, g in df.groupby("q"):
    for v in VARS:
        T = theil(g[v], g["Población Adulta"])
        reg = g.groupby("Region").apply(
            lambda h: pd.Series({"y": np.average(h[v], weights=h["Población Adulta"]),
                                 "w": h["Población Adulta"].sum()}))
        Tb = theil(reg["y"], reg["w"])
        cv = g[v].std() / g[v].mean()
        rows.append({"q": q, "var": v, "Theil": T, "Between": Tb,
                     "Within": T - Tb, "CV": cv})
ineq = pd.DataFrame(rows)
ineq["date"] = ineq["q"].dt.to_timestamp(how="end")
ineq.assign(var=ineq["var"].map(LONG)).drop(columns="date").round(5).to_csv(
    TAB / "t2_desigualdad_theil.csv", index=False)

fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.2), sharex=True)
for ax, v in zip(axes.flat, VARS):
    s = ineq[ineq["var"] == v].set_index("date")
    ax.fill_between(s.index, 0, s["Between"], color=BLUE_DARK, lw=0, label="Between regions")
    ax.fill_between(s.index, s["Between"], s["Theil"], color="#9ec5f4", lw=0,
                    label="Within regions")
    ax.set_ylim(0, None)
    share = (s["Between"] / s["Theil"]).iloc[-1]
    ax.set_title(LONG[v], fontsize=8.5, pad=14)
    ax.text(0, 1.02, f"between-region share 2026Q1: {share:.0%}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=6.5, color=INK2)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(mpl.dates.YearLocator(2))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
handles = [Patch(color=BLUE_DARK, label="Between-region component"),
           Patch(color="#9ec5f4", label="Within-region component")]
fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
fig.suptitle("Territorial inequality in access infrastructure (population-weighted Theil T)",
             x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout(rect=(0, 0.02, 1, 0.97))
note(fig, SOURCE + " Six economic regions as defined by CNBV.", y=-0.045)
save(fig, "fig02_desigualdad_theil")

# ----------------------------------------------------------------------------
# 3. K-means: model selection
# ----------------------------------------------------------------------------
print("K-means model selection…")
scaler = StandardScaler().fit(df[VARS])
Z = scaler.transform(df[VARS])

rng = np.random.default_rng(SEED)
sel = []
for k in range(2, 11):
    km = KMeans(k, init="k-means++", n_init=10, max_iter=300, random_state=SEED).fit(Z)
    aris = []
    for b in range(50):
        idx = rng.choice(len(Z), len(Z), replace=True)
        kb = KMeans(k, n_init=10, random_state=b).fit(Z[idx])
        aris.append(adjusted_rand_score(km.labels_, kb.predict(Z)))
    sel.append({"k": k, "inertia": km.inertia_,
                "silhouette": silhouette_score(Z, km.labels_),
                "calinski_harabasz": calinski_harabasz_score(Z, km.labels_),
                "davies_bouldin": davies_bouldin_score(Z, km.labels_),
                "bootstrap_ARI_mean": np.mean(aris), "bootstrap_ARI_sd": np.std(aris)})
sel = pd.DataFrame(sel)
sel.round(4).to_csv(TAB / "t3_seleccion_k.csv", index=False)

fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.3))
specs = [("inertia", "Within-cluster SS (elbow)", None),
         ("silhouette", "Silhouette (higher = better)", "max"),
         ("davies_bouldin", "Davies–Bouldin (lower = better)", "min"),
         ("bootstrap_ARI_mean", "Bootstrap stability (ARI)", None)]
for ax, (col, title, best) in zip(axes, specs):
    ax.plot(sel["k"], sel[col], color=BLUE, lw=1.8, marker="o", ms=4.5,
            mfc=SURF, mec=BLUE, mew=1.5)
    if col == "bootstrap_ARI_mean":
        ax.fill_between(sel["k"], sel[col] - sel["bootstrap_ARI_sd"],
                        sel[col] + sel["bootstrap_ARI_sd"], color="#cde2fb", lw=0)
    r = sel[sel.k == K].iloc[0]
    ax.plot(K, r[col], marker="o", ms=7, color=BLUE_DARK, mec=SURF, mew=1.5, zorder=5)
    ax.annotate(f"K={K}", (K, r[col]), textcoords="offset points", xytext=(6, 6),
                fontsize=7.5, color=INK)
    ax.set_title(title, fontsize=8.5)
    ax.set_xticks(range(2, 11))
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Number of clusters K", fontsize=7.5)
fig.tight_layout()
note(fig, "Note: K-means++ with 10 initialisations, random_state=42; stability = mean ARI over 50 "
          "bootstrap resamples (band: ±1 s.d.). " + SOURCE, y=-0.04)
save(fig, "fig03_seleccion_k")

# ----------------------------------------------------------------------------
# 4. Final K-means (K=7) and Rogers ordering
# ----------------------------------------------------------------------------
print("Final K-means…")
km = KMeans(K, init="k-means++", n_init=10, max_iter=300, random_state=SEED).fit(Z)
df["km_raw"] = km.labels_
cent_z = pd.DataFrame(km.cluster_centers_, columns=VARS)
cent = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=VARS)

# Digital Access Index (DAI): mean z-score of the five alternative/digital channels.
cent_z["DAI"] = cent_z[DIGITAL].mean(axis=1)
order = cent_z["DAI"].sort_values().index.tolist()      # lowest -> highest
STAGES = ["Entrenched laggards", "Laggards in transition", "Late majority",
          "Early majority", "Early adopters", "Innovators II", "Innovators I (frontier)"]
STAGES_ES = ["Rezagados tardíos", "Rezagados tempranos", "Mayoría tardía",
             "Mayoría temprana", "Primeros adoptantes", "Innovadores tardíos",
             "Innovadores tempranos"]
ROGERS = ["Laggards", "Laggards", "Late majority", "Early majority",
          "Early adopters", "Innovators", "Innovators"]
remap = {old: new for new, old in enumerate(order)}
df["stage"] = df["km_raw"].map(remap)                   # 0 = most lagging, 6 = frontier
df["stage_name"] = df["stage"].map(dict(enumerate(STAGES)))

cent = cent.loc[order].reset_index(drop=True)
cent_z = cent_z.loc[order].reset_index(drop=True)
cent_out = cent.rename(columns=LONG).copy()
cent_out.insert(0, "Stage", STAGES)
cent_out.insert(1, "Etapa (tesis)", STAGES_ES)
cent_out.insert(2, "Rogers category", ROGERS)
cent_out["DAI (z)"] = cent_z["DAI"].values
cent_out["n obs"] = df["stage"].value_counts().sort_index().values
cent_out["share obs"] = cent_out["n obs"] / len(df)
cent_out.round(3).to_csv(TAB / "t4_centroides_k7.csv", index=False)
df[["Fecha", "qlab", "Region", "Estado", "stage", "stage_name"] + VARS].to_csv(
    TAB / "t5_asignacion_estado_trimestre.csv", index=False)

# Membership spells per state (analogue of thesis Table 7)
spells = []
for (st, s), g in df.sort_values("q").groupby(["Estado", "stage"]):
    qs = g["q"].tolist()
    start = prev = qs[0]
    for qq in qs[1:] + [None]:
        if qq is None or qq != prev + 1:
            spells.append({"Stage": STAGES[s], "State": st, "from": str(start), "to": str(prev),
                           "quarters": (prev - start).n + 1})
            if qq is not None:
                start = qq
        prev = qq if qq is not None else prev
pd.DataFrame(spells).sort_values(["Stage", "State", "from"]).to_csv(
    TAB / "t6_permanencia_por_etapa.csv", index=False)

print(cent_out[["Stage", "Rogers category", "DAI (z)", "n obs"]].to_string(index=False))

# Fig 4 — centroid heatmap: colour = column-wise relative level, text = original units
fig, ax = plt.subplots(figsize=(7.2, 3.4))
rel = (cent - cent.min()) / (cent.max() - cent.min())
ax.imshow(rel.values, cmap=SEQ_BLUES, aspect="auto", vmin=0, vmax=1)
fmt = {"SBC_x10K": "{:.2f}", "SBD_x10K": "{:.2f}", "SSoc_x10K": "{:.2f}", "SSof_x10K": "{:.2f}",
       "Corr_x10K": "{:.1f}", "CA_x10K": "{:.1f}", "TPV_x10K": "{:.0f}", "ETPV_x10K": "{:.0f}",
       "CBM_x10K": "{:,.0f}"}
for i in range(K):
    for j, v in enumerate(VARS):
        val = cent.iloc[i, j]
        ax.text(j, i, fmt[v].format(val), ha="center", va="center", fontsize=7.5,
                color=SURF if rel.iloc[i, j] > 0.55 else INK)
ax.set_xticks(range(len(VARS)))
ax.set_xticklabels([textwrap.fill(LONG[v], 11) for v in VARS], fontsize=7)
ax.set_yticks(range(K))
ax.set_yticklabels([f"{s}  (n={n})" for s, n in zip(STAGES, cent_out["n obs"])], fontsize=7.5)
ax.grid(False)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_xticks(np.arange(-.5, len(VARS)), minor=True)
ax.set_yticks(np.arange(-.5, K), minor=True)
ax.grid(which="minor", color=SURF, lw=2)
ax.tick_params(which="minor", length=0)
ax.set_title("K-means centroids (K=7), access points per 10,000 adults", loc="left")
cb = fig.colorbar(mpl.cm.ScalarMappable(cmap=SEQ_BLUES), ax=ax, fraction=0.025, pad=0.01)
cb.set_ticks([0, 1]); cb.set_ticklabels(["min", "max"]); cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=7)
note(fig, "Note: colour is the column-wise relative level (min–max across clusters); figures are "
          "de-standardised centroids. Stages ordered by the Digital Access Index. " + SOURCE)
save(fig, "fig04_heatmap_centroides")

# Fig 5 — standardised profile (diverging), compact "fingerprint"
fig, ax = plt.subplots(figsize=(7.2, 3.2))
lim = 3
ax.imshow(cent_z[VARS].clip(-lim, lim).values, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
for i in range(K):
    for j, v in enumerate(VARS):
        z = cent_z.loc[i, v]
        ax.text(j, i, f"{z:+.1f}", ha="center", va="center", fontsize=7.5,
                color=SURF if abs(z) > 1.6 else INK)
ax.set_xticks(range(len(VARS)))
ax.set_xticklabels([SHORT[v] for v in VARS], fontsize=7.5)
ax.set_yticks(range(K)); ax.set_yticklabels(STAGES, fontsize=7.5)
ax.grid(False)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_xticks(np.arange(-.5, len(VARS)), minor=True)
ax.set_yticks(np.arange(-.5, K), minor=True)
ax.grid(which="minor", color=SURF, lw=2); ax.tick_params(which="minor", length=0)
ax.set_title("Standardised centroid profiles (z-scores)", loc="left")
cb = fig.colorbar(mpl.cm.ScalarMappable(cmap=DIVERGING, norm=mpl.colors.Normalize(-lim, lim)),
                  ax=ax, fraction=0.025, pad=0.01)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
cb.set_label("z-score (negative = below national mean)", fontsize=7, color=INK2)
note(fig, "SBC/SBD/SSoc/SSof: commercial, development-bank, Socap and Sofipo branches; Corr: banking "
          "agents; CA: ATMs; TPV: POS terminals; ETPV: merchants with POS; CBM: mobile-banking contracts.")
save(fig, "fig05_perfiles_z")

# Fig 6 — state × quarter stage map (the "trajectory map")
REG_ORDER = ["Mexico City", "Northwest", "Northeast", "West & Bajío", "Centre-South & East", "South"]
mean_stage = df.groupby("Estado")["stage"].mean()
st_reg = df.groupby("Estado")["Region"].first()
states_sorted = (pd.DataFrame({"reg": st_reg, "m": mean_stage})
                 .assign(r=lambda d: d.reg.map({r: i for i, r in enumerate(REG_ORDER)}))
                 .sort_values(["r", "m"], ascending=[True, False]).index.tolist())
mat = df.pivot(index="Estado", columns="qlab", values="stage").loc[states_sorted]
cmap7 = ListedColormap(RAMP7)
fig, ax = plt.subplots(figsize=(7.4, 7.0))
ax.imshow(mat.values, cmap=cmap7, norm=BoundaryNorm(np.arange(-.5, 7.5), 7), aspect="auto")
ax.set_xticks(np.arange(-.5, mat.shape[1]), minor=True)
ax.set_yticks(np.arange(-.5, mat.shape[0]), minor=True)
ax.grid(which="minor", color=SURF, lw=1.2); ax.grid(which="major", visible=False)
ax.tick_params(which="minor", length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_yticks(range(len(states_sorted))); ax.set_yticklabels(states_sorted, fontsize=7.2)
xt = [i for i, c in enumerate(mat.columns) if c.endswith("Q1")]
ax.set_xticks(xt); ax.set_xticklabels([mat.columns[i][:4] for i in xt], fontsize=7.5)
ax.xaxis.tick_top()
# region separators & labels
regs = [st_reg[s] for s in states_sorted]
for i in range(1, len(regs)):
    if regs[i] != regs[i - 1]:
        ax.axhline(i - .5, color=INK, lw=1.0)
starts = [0] + [i for i in range(1, len(regs)) if regs[i] != regs[i - 1]]
ends = starts[1:] + [len(regs)]
for s, e in zip(starts, ends):
    ax.text(mat.shape[1] + 0.6, (s + e - 1) / 2, regs[s], va="center", ha="left",
            fontsize=7.2, color=INK2)
for x, lab in [(QUARTERS.index(pd.Period("2020Q1")), "COVID-19"),
               (QUARTERS.index(pd.Period("2022Q1")), "BDIF v3")]:
    ax.axvline(x - .5, color=INK, lw=0.8, ls=(0, (2, 2)))
    ax.text(x - .4, len(states_sorted) - .2, lab, fontsize=6.8, color=INK2, va="top")
_h = [Patch(facecolor=c, edgecolor=AXIS, lw=0.5, label=s) for c, s in zip(RAMP7, STAGES)][::-1]
_h = [_h[i] for i in (0, 4, 1, 5, 2, 6, 3)]          # row-major reading order, 4 columns
ax.legend(handles=_h,
          loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=7.2,
          handlelength=1.2, columnspacing=1.0)
ax.set_title("Adoption stage of each state, quarter by quarter (K-means, K=7)", loc="left", pad=22)
note(fig, SOURCE, y=0.02)
save(fig, "fig06_trayectorias_estado")

# Fig 7 — share of adult population living in each stage
pop = (df.groupby(["date", "stage"])["Población Adulta"].sum().unstack(fill_value=0))
share = pop.div(pop.sum(axis=1), axis=0)
share.rename(columns=dict(enumerate(STAGES))).round(4).to_csv(TAB / "t7_poblacion_por_etapa.csv")
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.stackplot(share.index, *[share[s] for s in range(K)], colors=RAMP7,
             edgecolor=SURF, linewidth=0.8)
ax.set_ylim(0, 1); ax.set_xlim(share.index.min(), share.index.max())
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
ax.grid(axis="x", visible=False)
ax.axvline(pd.Timestamp("2020-03-31"), color=INK, lw=0.8, ls=(0, (2, 2)))
ax.axvline(pd.Timestamp("2022-03-31"), color=INK, lw=0.8, ls=(0, (2, 2)))
last = share.iloc[-1].cumsum()
for s in range(K):
    if share.iloc[-1, s] > 0.025:
        mid = last[s] - share.iloc[-1, s] / 2
        ax.annotate(f"{STAGES[s]}  {share.iloc[-1, s]:.0%}", (share.index[-1], mid),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=7.2,
                    color=INK2)
ax.set_title("Share of Mexico's adult population living in each adoption stage", loc="left")
ax.set_ylabel("Adult population")
note(fig, "Dashed lines: COVID-19 onset (2020Q1) and BDIF version 3 (2022Q1). " + SOURCE)
save(fig, "fig07_poblacion_por_etapa")

# ----------------------------------------------------------------------------
# 5. Transition matrices by period
# ----------------------------------------------------------------------------
print("Transition matrices…")
df = df.sort_values(["Estado", "q"])
df["stage_next"] = df.groupby("Estado")["stage"].shift(-1)
tr = df.dropna(subset=["stage_next"]).copy()
tr["stage_next"] = tr["stage_next"].astype(int)
tr["period"] = tr["q"].apply(lambda q: period(q + 1))
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True)
mob = []
for ax, p in zip(axes, PERIODS):
    t = pd.crosstab(tr.loc[tr.period == p, "stage"], tr.loc[tr.period == p, "stage_next"])
    t = t.reindex(index=range(K), columns=range(K), fill_value=0)
    P = t.div(t.sum(axis=1).replace(0, np.nan), axis=0)
    n_moves = (t.values.sum() - np.trace(t.values))
    up = np.triu(t.values, 1).sum(); down = np.tril(t.values, -1).sum()
    mob.append({"period": p, "transitions": int(t.values.sum()), "moves": int(n_moves),
                "upward": int(up), "downward": int(down),
                "mobility_rate": n_moves / t.values.sum()})
    P.rename(index=dict(enumerate(STAGES)), columns=dict(enumerate(STAGES))).round(3).to_csv(
        TAB / f"t8_transicion_{p.split(' ')[0].lower()}.csv")
    ax.imshow(P.fillna(0).values, cmap=SEQ_BLUES, vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            val = P.iloc[i, j]
            if pd.notna(val) and t.iloc[i, j] > 0:
                ax.text(j, i, f"{val:.2f}" if val < 1 else "1", ha="center", va="center",
                        fontsize=6, color=SURF if val > 0.55 else INK)
    ax.set_xticks(range(K)); ax.set_xticklabels(range(1, K + 1), fontsize=7)
    ax.set_yticks(range(K)); ax.set_yticklabels([f"{i+1} {s}" for i, s in enumerate(STAGES)],
                                                fontsize=6.8)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-.5, K), minor=True); ax.set_yticks(np.arange(-.5, K), minor=True)
    ax.grid(which="minor", color=SURF, lw=1.5); ax.tick_params(which="minor", length=0)
    ax.set_title(f"{p}\n{up} up · {down} down", fontsize=8)
    ax.set_xlabel("Stage at t+1", fontsize=7.5)
axes[0].set_ylabel("Stage at t", fontsize=7.5)
fig.tight_layout()
note(fig, "Note: quarter-to-quarter transition probabilities; empty cells = no observed transitions. "
     + SOURCE, y=-0.03)
save(fig, "fig08_matrices_transicion")
pd.DataFrame(mob).round(4).to_csv(TAB / "t9_movilidad.csv", index=False)

# ----------------------------------------------------------------------------
# 6. DBSCAN (density-based) robustness check
# ----------------------------------------------------------------------------
print("DBSCAN…")
pca = PCA(n_components=3, random_state=SEED).fit(Z)
PCs = pca.transform(Z)
pd.DataFrame(pca.components_.T, index=[LONG[v] for v in VARS],
             columns=["PC1", "PC2", "PC3"]).round(3).assign(
    **{"": ""}).to_csv(TAB / "t10_pca_cargas.csv")
MINPTS = 10
kd = np.sort(NearestNeighbors(n_neighbors=MINPTS).fit(PCs).kneighbors(PCs)[0][:, -1])
# knee = max distance to the chord of the sorted k-distance curve
x = np.arange(len(kd)); x_n = x / x.max(); y_n = (kd - kd.min()) / (kd.max() - kd.min())
knee = int(np.argmax(np.abs(y_n - x_n)))
EPS = float(kd[knee])
db = DBSCAN(eps=EPS, min_samples=MINPTS).fit(PCs)
df = df.sort_index()
df["db"] = db.labels_
n_db = len(set(db.labels_)) - (-1 in db.labels_)
print(f"  eps={EPS:.3f} (knee), minPts={MINPTS}: {n_db} clusters, noise={np.mean(db.labels_==-1):.1%}")
core_mask = db.labels_ >= 0
db_stats = {
    "eps": EPS, "min_samples": MINPTS, "pca_var_explained": pca.explained_variance_ratio_.sum(),
    "n_clusters": n_db, "noise_share": np.mean(db.labels_ == -1),
    "ARI_vs_kmeans": adjusted_rand_score(df["stage"], df["db"]),
    "silhouette_nonnoise": silhouette_score(PCs[core_mask], db.labels_[core_mask])
    if n_db > 1 else np.nan,
}
pd.Series(db_stats).round(4).to_csv(TAB / "t11_dbscan_resumen.csv", header=["value"])
xt_db = pd.crosstab(df["db"], df["stage_name"]).reindex(columns=STAGES, fill_value=0)
xt_db.to_csv(TAB / "t12_dbscan_vs_kmeans.csv")
db_members = (df.groupby(["db", "Estado"]).size().rename("quarters").reset_index()
              .sort_values(["db", "quarters"], ascending=[True, False]))
db_members.to_csv(TAB / "t13_dbscan_miembros.csv", index=False)
df[["qlab", "Estado", "stage", "db"]].to_csv(TAB / "t20_dbscan_estado_trimestre.csv", index=False)

# name DBSCAN groups from their membership (data-driven, checked against t13)
def _db_name(lab):
    if lab == -1:
        return "Noise / outliers"
    mem = df.loc[df.db == lab, "Estado"].value_counts()
    if lab == df.loc[df.db >= 0, "db"].value_counts().idxmax():
        return "Dense core (centre, south, Bajío)"
    if {"Nuevo León", "Quintana Roo"} <= set(mem.index):
        return "Metropolitan-tourist frontier"
    if {"Chihuahua", "Sonora"} <= set(mem.index):
        return "Northern border belt"
    return f"Micro-group: {mem.index[0]}"


db_names = {lab: _db_name(lab) for lab in sorted(df["db"].unique())}
_palette = {"Dense core (centre, south, Bajío)": "#9ec5f4",
            "Northern border belt": "#2a78d6",
            "Metropolitan-tourist frontier": "#0d366b",
            "Noise / outliers": "#898781"}
_micro = iter(["#5598e7", "#184f95", "#6da7ec", "#256abf"])
db_cols = {lab: _palette.get(n) or next(_micro) for lab, n in db_names.items()}
pd.Series(db_names).to_csv(TAB / "t13b_dbscan_nombres.csv", header=["name"])

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [0.9, 1.2, 1.2]})
ax = axes[0]
ax.plot(x, kd, color=BLUE, lw=1.6)
ax.axhline(EPS, color=INK, lw=0.8, ls=(0, (2, 2)))
ax.plot(knee, EPS, "o", color=BLUE_DARK, ms=6, mec=SURF, mew=1.2)
ax.annotate(f"ε = {EPS:.2f}", (knee, EPS), xytext=(-60, 12), textcoords="offset points",
            fontsize=7.5)
ax.set_title(f"{MINPTS}-NN distance (sorted)", fontsize=8.5)
ax.set_xlabel("Observations", fontsize=7.5); ax.tick_params(labelsize=7)
for ax, (title, labels, cols, names) in zip(axes[1:], [
        ("K-means stages", df["stage"].values, dict(enumerate(RAMP7)), dict(enumerate(STAGES))),
        ("DBSCAN groups", df["db"].values, db_cols, db_names)]):
    for lab in sorted(set(labels), key=lambda l: (l == -1, -(labels == l).sum())):
        m = labels == lab
        ax.scatter(PCs[m, 0], PCs[m, 1], s=9 if lab != -1 else 14, c=cols[lab],
                   edgecolors=SURF if lab != -1 else "none", linewidths=0.3,
                   marker="o" if lab != -1 else "X",
                   label=f"{names[lab]} ({m.sum()})", alpha=0.95)
    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})", fontsize=7.5)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})", fontsize=7.5)
    ax.tick_params(labelsize=7)
axes[2].legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1.0), markerscale=1.6,
               handletextpad=0.2, labelspacing=0.8)
fig.tight_layout()
note(fig, f"Note: DBSCAN on the first three principal components ({pca.explained_variance_ratio_.sum():.0%} "
          f"of variance), minPts={MINPTS}, ε at the knee of the k-distance curve. "
          f"ARI vs K-means = {db_stats['ARI_vs_kmeans']:.2f}. K-means stage colours as in Fig. 6 "
          "(light = entrenched laggards, dark = frontier). " + SOURCE, y=-0.04)
save(fig, "fig09_dbscan")

# Fig 10 — PCA map with state trajectories (selected states)
HIGHLIGHT = ["Ciudad de México", "Nuevo León", "Querétaro", "Jalisco", "Oaxaca", "Chiapas", "México"]
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for s in range(K):
    m = df["stage"].values == s
    ax.scatter(PCs[m, 0], PCs[m, 1], s=10, c=RAMP7[s], edgecolors="none", alpha=0.55,
               label=STAGES[s])
for st in HIGHLIGHT:
    m = (df["Estado"] == st).values
    idx = np.where(m)[0][np.argsort(df.loc[m, "q"].values)]
    ax.plot(PCs[idx, 0], PCs[idx, 1], color=INK, lw=0.9, alpha=0.8)
    ax.annotate("", xy=PCs[idx[-1], :2], xytext=PCs[idx[-2], :2],
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, mutation_scale=8))
    ax.scatter(*PCs[idx[0], :2], s=14, color=SURF, edgecolors=INK, linewidths=0.9, zorder=4)
    ax.annotate(st, PCs[idx[-1], :2], xytext=(5, 3), textcoords="offset points",
                fontsize=7.3, color=INK, fontweight="bold")
ax.set_xlabel(f"PC1 — breadth of digital/alternative access ({pca.explained_variance_ratio_[0]:.0%})")
ax.set_ylabel(f"PC2 — popular finance vs. public/mobile ({pca.explained_variance_ratio_[1]:.0%})")
leg = ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4,
                markerscale=1.8, handletextpad=0.2, columnspacing=1.2)
for h in leg.legend_handles:
    h.set_alpha(1)
ax.set_title("Where states sit and how they moved, 2018Q1 to 2026Q1 (principal-component map)",
             loc="left")
note(fig, "Note: open circle = 2018Q1, arrow = 2026Q1. " + SOURCE, y=-0.1)
save(fig, "fig10_mapa_pca_trayectorias")

# ----------------------------------------------------------------------------
# 7. Exploratory event study: Banco del Bienestar expansion (place-based policy)
# ----------------------------------------------------------------------------
print("Event study (Banco del Bienestar)…")
base_q, end_q = pd.Period("2021Q4"), pd.Period("2024Q4")
sbd = df.pivot(index="Estado", columns="q", values="SBD_x10K")
intensity = (sbd[end_q] - sbd[base_q]).rename("dSBD")
intensity.sort_values(ascending=False).round(4).to_csv(TAB / "t14_intensidad_bienestar.csv")
df = df.merge(intensity, left_on="Estado", right_index=True)
df["I"] = (df["dSBD"] - df["dSBD"].mean()) / df["dSBD"].std()
df["rel"] = df["q"].apply(lambda q: (q - base_q).n)
df["st_id"] = df["Estado"].astype("category").cat.codes
OUTCOMES = {"CBM_x10K": "Mobile-banking contracts", "Corr_x10K": "Banking agents",
            "CA_x10K": "ATMs", "SBC_x10K": "Commercial bank branches"}
es_rows, did_rows = [], []
for y in OUTCOMES:
    d = df.copy()
    d["ly"] = np.log(d[y])
    rel_vals = sorted(d["rel"].unique())
    terms = []
    for r in rel_vals:
        if r == 0:
            continue
        nm = f"e_{'m' if r < 0 else 'p'}{abs(r)}"
        d[nm] = (d["rel"] == r) * d["I"]
        terms.append((r, nm))
    f = "ly ~ " + " + ".join(n for _, n in terms) + " + C(Estado) + C(qlab)"
    res = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["st_id"]})
    for r, nm in terms:
        es_rows.append({"outcome": y, "rel": r, "beta": res.params[nm],
                        "lo": res.conf_int().loc[nm, 0], "hi": res.conf_int().loc[nm, 1]})
    es_rows.append({"outcome": y, "rel": 0, "beta": 0, "lo": 0, "hi": 0})
    pre = [n for r, n in terms if r < 0]
    ftest = res.f_test(" = ".join(pre[:1]) + " = 0" if len(pre) == 1 else
                       ", ".join(f"{n} = 0" for n in pre))
    d["post_I"] = (d["rel"] > 0) * d["I"]
    r2 = smf.ols("ly ~ post_I + C(Estado) + C(qlab)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["st_id"]})
    did_rows.append({"outcome": OUTCOMES[y], "beta_post×I": r2.params["post_I"],
                     "se": r2.bse["post_I"], "p": r2.pvalues["post_I"],
                     "pretrend_F_p": float(ftest.pvalue), "n": int(r2.nobs)})
es = pd.DataFrame(es_rows).sort_values(["outcome", "rel"])
es.round(5).to_csv(TAB / "t15_event_study.csv", index=False)
did = pd.DataFrame(did_rows)
did.round(4).to_csv(TAB / "t16_did_bienestar.csv", index=False)
print(did.round(4).to_string(index=False))

fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex=True)
for ax, y in zip(axes.flat, OUTCOMES):
    e = es[es.outcome == y]
    xq = [(base_q + int(r)).to_timestamp(how="end") for r in e["rel"]]
    ax.fill_between(xq, e["lo"], e["hi"], color="#cde2fb", lw=0)
    ax.plot(xq, e["beta"], color=BLUE, lw=1.6, marker="o", ms=2.8)
    ax.axhline(0, color=AXIS, lw=0.9)
    ax.axvline(base_q.to_timestamp(how="end"), color=INK, lw=0.8, ls=(0, (2, 2)))
    r = did[did.outcome == OUTCOMES[y]].iloc[0]
    ax.set_title(OUTCOMES[y], fontsize=8.5, pad=14)
    ax.text(0, 1.02, f"post × intensity {r['beta_post×I']:+.3f} (p={r['p']:.2f}) · "
                     f"pre-trend F-test p={r['pretrend_F_p']:.3f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5, color=INK2)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(mpl.dates.YearLocator(2))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
for ax in axes[:, 0]:
    ax.set_ylabel("Effect on log outcome\nper 1 s.d. of intensity", fontsize=7.5)
fig.suptitle("Exploratory event study: Banco del Bienestar branch expansion (2021Q4 = reference)",
             x=0.01, ha="left", fontsize=10.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96))
note(fig, "Note: TWFE regression of log outcome on state and quarter fixed effects plus quarter × "
          "standardised change in development-bank branches per 10,000 adults (2021Q4 to 2024Q4); 95% CI clustered by state (32 clusters). Descriptive, "
          "not causal: branch placement was targeted. " + SOURCE, y=-0.01)
save(fig, "fig11_event_study_bienestar")

# ----------------------------------------------------------------------------
# 7b. Robustness: K-means on within-quarter standardised data (relative positions)
# ----------------------------------------------------------------------------
print("Robustness (within-quarter z-scores)…")
df = df.sort_values(["q", "Estado"]).reset_index(drop=True)
Zq = df.groupby("q")[VARS].transform(lambda x: (x - x.mean()) / x.std()).values
km_q = KMeans(K, init="k-means++", n_init=10, random_state=SEED).fit(Zq)
dai_q = pd.DataFrame(km_q.cluster_centers_, columns=VARS)[DIGITAL].mean(axis=1).sort_values()
df["stage_rel"] = pd.Series(km_q.labels_).map({o: i for i, o in enumerate(dai_q.index)}).values
lag0 = df.loc[(df.q == QUARTERS[0]) & (df.stage == 0), "Estado"]
_t = df.sort_values(["Estado", "q"]).assign(
    nxt=lambda d: d.groupby("Estado")["stage_rel"].shift(-1)).dropna(subset=["nxt"])
rob = {
    "silhouette_within_quarter": silhouette_score(Zq, km_q.labels_),
    "ARI_vs_pooled": adjusted_rand_score(df["stage"], df["stage_rel"]),
    "relative_moves": int((_t["stage_rel"] != _t["nxt"]).sum()),
    "relative_transitions": int(len(_t)),
    "laggards2018_in_bottom2_relative_2026Q1":
        int(df[(df.q == QUARTERS[-1]) & df.Estado.isin(lag0)]["stage_rel"].le(1).sum()),
    "laggards2018_n": int(len(lag0)),
}
pd.Series(rob).to_csv(TAB / "t18_robustez_relativa.csv", header=["value"])
df.pivot(index="Estado", columns="qlab", values="stage_rel").to_csv(
    TAB / "t19_etapa_relativa_estado_trimestre.csv")
print(pd.Series(rob).round(3).to_string())

# ----------------------------------------------------------------------------
# 7c. Sensitivity: alternative K, log transform, DBSCAN parameter grid
# ----------------------------------------------------------------------------
print("Sensitivity analyses…")
Zp = scaler.transform(df[VARS])                    # df is now sorted by q, Estado
pop_last = df[df.q == QUARTERS[-1]]
pop_first = df[df.q == QUARTERS[0]]


def ordered_labels(Zm, k, seed=SEED):
    m = KMeans(k, init="k-means++", n_init=10, random_state=seed).fit(Zm)
    dai = pd.DataFrame(m.cluster_centers_, columns=VARS)[DIGITAL].mean(axis=1).sort_values()
    return pd.Series(m.labels_).map({o: i for i, o in enumerate(dai.index)}).values, m


sens = []
specs = [(k, "z-score", Zp) for k in (5, 6, 7, 8)] + \
        [(7, "log + z-score", StandardScaler().fit_transform(np.log1p(df[VARS])))]
for k, tr, Zm in specs:
    lab, m = ordered_labels(Zm, k)
    bottom = lab <= 1
    sel_first = (df.q == QUARTERS[0]).values
    sel_last = (df.q == QUARTERS[-1]).values
    w = df["Población Adulta"].values
    lag0_mask = df["Estado"].isin(lag0).values
    sens.append({
        "K": k, "transform": tr,
        "silhouette": silhouette_score(Zm, m.labels_),
        "ARI_vs_main": adjusted_rand_score(df["stage"], lab),
        "initial_laggards_in_bottom2_2018Q1": int((bottom & sel_first & lag0_mask).sum()),
        "initial_laggards_in_bottom2_2026Q1": int((bottom & sel_last & lag0_mask).sum()),
        "adult_share_bottom2_2018Q1": w[bottom & sel_first].sum() / w[sel_first].sum(),
        "adult_share_bottom2_2026Q1": w[bottom & sel_last].sum() / w[sel_last].sum(),
        "upward_moves_post2022": int(pd.DataFrame({"e": df["Estado"], "q": df["q"], "l": lab})
                                     .sort_values(["e", "q"]).assign(n=lambda d: d.groupby("e")["l"].shift(-1))
                                     .query("q >= @pd.Period('2021Q4')").pipe(lambda d: (d.n > d.l).sum())),
    })
sens = pd.DataFrame(sens)
# relative-position check on log data (within-quarter z of log values)
Zlq = np.log1p(df[VARS]).groupby(df["q"]).transform(lambda x: (x - x.mean()) / x.std()).values
lab_lq, _m = ordered_labels(Zlq, 7)
_last = (df.q == QUARTERS[-1]).values
sens["relative_log_initial_laggards_in_bottom2_2026Q1"] = np.nan
sens.loc[sens["transform"] == "log + z-score", "relative_log_initial_laggards_in_bottom2_2026Q1"] = int(
    ((lab_lq <= 1) & _last & df["Estado"].isin(lag0).values).sum())
sens["bootstrap_ARI"] = sens["K"].map(sel.set_index("k")["bootstrap_ARI_mean"]).where(
    sens["transform"] == "z-score")
sens.round(4).to_csv(TAB / "t21_sensibilidad_kmeans.csv", index=False)
print(sens.round(3).to_string(index=False))

north = ["Aguascalientes", "Baja California", "Chihuahua", "Coahuila", "Sinaloa", "Sonora", "Tamaulipas"]
grid = []
for mp in (6, 8, 10, 12, 15):
    kdm = np.sort(NearestNeighbors(n_neighbors=mp).fit(PCs).kneighbors(PCs)[0][:, -1])
    xn = np.arange(len(kdm)) / (len(kdm) - 1); yn = (kdm - kdm.min()) / (kdm.max() - kdm.min())
    e0 = float(kdm[int(np.argmax(np.abs(yn - xn)))])
    for mult in (0.8, 1.0, 1.2):
        lab = DBSCAN(eps=e0 * mult, min_samples=mp).fit_predict(PCs)
        ncl = len(set(lab)) - (-1 in lab)
        core = pd.Series(lab[lab >= 0]).value_counts().idxmax() if ncl else -99
        nmask = df["Estado"].isin(north).values
        nlabs = pd.Series(lab[nmask]).value_counts()
        north_separate = bool(ncl > 1 and nlabs.index[0] not in (-1, core)
                              and nlabs.iloc[0] / nmask.sum() > 0.9)
        cdmx = (df["Estado"] == "Ciudad de México").values
        grid.append({"minPts": mp, "eps": e0 * mult, "eps_rule": f"{mult:.1f} × knee",
                     "clusters": ncl, "noise_share": np.mean(lab == -1),
                     "core_share": np.mean(lab == core) if ncl else 0,
                     "north_belt_separate": north_separate,
                     "cdmx_noise_share": np.mean(lab[cdmx] == -1),
                     "ARI_vs_kmeans": adjusted_rand_score(df["stage"], lab)})
grid = pd.DataFrame(grid)
grid.round(4).to_csv(TAB / "t22_sensibilidad_dbscan.csv", index=False)
print(grid.round(3).to_string(index=False))

# ----------------------------------------------------------------------------
# 8. Key numbers for the text
# ----------------------------------------------------------------------------
first, lastq = df["q"].min(), df["q"].max()
nat = df.groupby("q").apply(lambda g: pd.Series(
    {v: g[v.replace("_x10K", "")].sum() / g["Población Adulta"].sum() * 1e4 for v in VARS}))
growth = (nat.loc[lastq] / nat.loc[first] - 1).rename("growth_2018Q1_2026Q1")
pd.concat([nat.loc[first].rename("2018Q1"), nat.loc[lastq].rename("2026Q1"), growth], axis=1) \
    .rename(index=LONG).round(3).to_csv(TAB / "t17_nacional_crecimiento.csv")
lag_share = share[[0, 1]].sum(axis=1)
key = {
    "obs": len(df), "states": df.Estado.nunique(), "quarters": len(QUARTERS),
    "silhouette_K7": sel.loc[sel.k == K, "silhouette"].item(),
    "bootstrapARI_K7": sel.loc[sel.k == K, "bootstrap_ARI_mean"].item(),
    "pop_share_laggards_2018Q1": lag_share.iloc[0],
    "pop_share_laggards_2026Q1": lag_share.iloc[-1],
    "pop_share_entrenched_2018Q1": share[0].iloc[0],
    "pop_share_entrenched_2026Q1": share[0].iloc[-1],
    "theil_CBM_first": ineq[(ineq["var"] == "CBM_x10K") & (ineq.q == first)].Theil.item(),
    "theil_CBM_last": ineq[(ineq["var"] == "CBM_x10K") & (ineq.q == lastq)].Theil.item(),
    "theil_SBD_first": ineq[(ineq["var"] == "SBD_x10K") & (ineq.q == first)].Theil.item(),
    "theil_SBD_last": ineq[(ineq["var"] == "SBD_x10K") & (ineq.q == lastq)].Theil.item(),
    **{f"dbscan_{k}": v for k, v in db_stats.items()},
}
pd.Series(key).to_csv(TAB / "t00_cifras_clave.csv", header=["value"])
print(pd.Series(key).round(4).to_string())
print("Done.")
