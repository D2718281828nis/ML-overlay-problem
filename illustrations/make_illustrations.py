"""Figures that explain why the unsupervised graph method (connected components of a
k-NN graph) wins on the chain datasets, and why graphs do not help on the others.

Run from anywhere:  python illustrations/make_illustrations.py
Writes PNG files next to this script. Uses the CSVs in the repository root.
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RED, GREEN, GREY, GOLD = "#d6363c", "#2f9a5c", "#9aa4b2", "#d99a00"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})


def load(name):
    d = pd.read_csv(os.path.join(ROOT, name))
    X = d[["X", "Y"]].to_numpy(float)
    y = (d.Color == "red").to_numpy().astype(int)
    return X, StandardScaler().fit_transform(X), y


def knn_graph(Z, k):
    A = kneighbors_graph(Z, k, mode="connectivity", include_self=False)
    return ((A + A.T) > 0).astype(float).tocsr()


def flag_top(score, n):
    lab = np.zeros(len(score), dtype=int)
    lab[np.argsort(-score)[:n]] = 1
    return lab


def f1_top(y, score):
    return f1_score(y, flag_top(score, y.sum()))


def comp_size(Z, k):
    nc, lab = connected_components(knn_graph(Z, k), directed=False)
    return np.bincount(lab)[lab].astype(float), lab, nc


def save(fig, name):
    fig.tight_layout(rect=[0, 0, 1, 0.9])     # keep room for the figure title above the panel titles
    fig.savefig(os.path.join(HERE, name), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


DATA = {"original": "df_red_green_overlay.csv", "ring": "df_red_green_distinct.csv",
        "chains (walk)": "df_red_green_chains.csv", "chains (circles)": "df_red_green_circles.csv"}
X, Z, y = load(DATA["chains (walk)"])
Xc, Zc, yc = load(DATA["chains (circles)"])

# ---------------------------------------------------------------- 01 overview
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
for ax, (XX, yy, title) in zip(axes, [(X, y, "Chains (walk): red = 6 long chains, green ≈ 95 short chains"),
                                      (Xc, yc, "Chains (circles): red = 6 large circles, green ≈ 90 small circles")]):
    ax.scatter(*XX[yy == 0].T, s=4, c=GREEN, label="green")
    ax.scatter(*XX[yy == 1].T, s=6, c=RED, label="red")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", markerscale=3)
fig.suptitle("01 · Red and green overlay: the same area, the same centre, the same spread", fontweight="bold")
save(fig, "01_datasets.png")

# ---------------------------------------------------------------- 02 locally identical
fig = plt.figure(figsize=(13, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.6])
nn = NearestNeighbors(n_neighbors=11).fit(X)
dist, _ = nn.kneighbors(X)
for j, (cls, col, name) in enumerate([(1, RED, "a red point"), (0, GREEN, "a green point")]):
    ax = fig.add_subplot(gs[0, j])
    i0 = np.where(y == cls)[0][len(np.where(y == cls)[0]) // 2 + 3]
    near = np.linalg.norm(X - X[i0], axis=1) < 70
    ax.scatter(*(X[near] - X[i0]).T, s=60, c=GREY, edgecolors="none")
    ax.scatter(0, 0, s=140, c=col, edgecolors="k", zorder=3)
    for q in np.argsort(np.linalg.norm(X - X[i0], axis=1))[1:3]:
        ax.plot([0, X[q, 0] - X[i0, 0]], [0, X[q, 1] - X[i0, 1]], c="k", lw=1.5)
    ax.add_patch(plt.Circle((0, 0), 30, fill=False, ls="--", ec=GREY))
    ax.set_xlim(-72, 72); ax.set_ylim(-72, 72); ax.set_aspect("equal")
    ax.set_title(f"Around {name}\n(grey = what a method sees)")
    ax.set_xticks([]); ax.set_yticks([])
ax = fig.add_subplot(gs[0, 2])
for k, ls in [(1, "-"), (2, "--"), (5, ":")]:
    for cls, col in [(1, RED), (0, GREEN)]:
        v = np.sort(dist[y == cls, k])
        ax.plot(v, np.linspace(0, 1, len(v)), c=col, ls=ls, lw=2, label=f"{'red' if cls else 'green'}, k = {k}")
ax.set_xlabel("distance to the k-th nearest neighbour (px)")
ax.set_ylabel("share of points")
ax.set_xlim(0, 80)
ax.legend(fontsize=8, ncol=3)
ax.set_title("Neighbour distances: almost identical\n(the only gap: chain ends, where the 2nd neighbour is 24 px away)")
fig.suptitle("02 · Locally every point looks the same: two neighbours 12 px away, nothing else within 30 px",
             fontweight="bold")
save(fig, "02_locally_identical.png")

# ---------------------------------------------------------------- 03 the k=2 graph = the chains
size, lab, nc = comp_size(Z, 2)
A = knn_graph(Z, 2)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
r, c = sparse.triu(A).nonzero()
segs = np.stack([X[r], X[c]], axis=1)
from matplotlib.collections import LineCollection
cmap = plt.get_cmap("tab20")
ax.add_collection(LineCollection(segs, colors=[cmap(lab[i] % 20) for i in r], linewidths=1.6))
ax.scatter(*X.T, s=3, c=[cmap(l % 20) for l in lab])
for comp in np.unique(lab[y == 1]):
    m = lab == comp
    cx, cy = X[m].mean(0)
    ax.annotate(f"{m.sum()} pts", (cx, cy), fontsize=9, fontweight="bold", color=RED,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=RED, alpha=.85))
ax.set_aspect("equal")
ax.set_title(f"k-NN graph with k = 2: {nc} connected components, one colour per component\n"
             "(red labels = the 6 red chains and their sizes)")
ax = axes[1]
sizes = np.bincount(lab)
red_comp = np.array([y[lab == i].mean() > .5 for i in range(nc)])
bins = np.arange(0, sizes.max() + 3, 2)
ax.hist(sizes[~red_comp], bins=bins, color=GREEN, label="components of green points")
ax.hist(sizes[red_comp], bins=bins, color=RED, label="components of red points")
ax.axvspan(sizes[~red_comp].max() + .5, sizes[red_comp].min() - .5, color=GOLD, alpha=.2,
           label="any threshold here gives F1 = 1")
ax.set_xlabel("component size (points)")
ax.set_ylabel("number of components")
ax.legend(fontsize=9)
ax.set_title("Component size separates the colours perfectly")
same = (y[r] == y[c]).mean()
fig.suptitle(f"03 · With k = 2 the graph's components ARE the chains ({same:.0%} of edges join the same colour, "
             "0 mixed components)", fontweight="bold")
save(fig, "03_graph_components.png")

# ---------------------------------------------------------------- 04 information has to travel far
n_red = y.sum()
D = shortest_path(knn_graph(Z, 2), unweighted=True, directed=False)
hops = np.arange(0, 21)
reach = np.array([(D <= h).sum(1) for h in hops])            # nodes reachable within h hops
f1_h = [f1_top(y, reach[i] + 1e-6 * np.random.default_rng(0).random(len(y))) for i in range(len(hops))]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
for cls, col in [(1, RED), (0, GREEN)]:
    med = np.median(reach[:, y == cls], axis=1)
    q1, q3 = np.percentile(reach[:, y == cls], [10, 90], axis=1)
    ax.plot(hops, med, c=col, lw=2.5, label=f"{'red' if cls else 'green'} points (median, 10–90%)")
    ax.fill_between(hops, q1, q3, color=col, alpha=.15)
ax.set_xlabel("hops along the graph (h)")
ax.set_ylabel("points reachable within h hops")
ax.legend()
ax.set_title("At 1–3 hops red and green see the same thing;\nonly far away does the red chain keep going")
ax = axes[1]
ax.plot(hops, f1_h, "o-", c="k", lw=2)
ax.axhline(2 * y.mean() / (1 + y.mean()), ls="--", c=GREY, label="all-red baseline")
ax.set_xlabel("hops along the graph (h)")
ax.set_ylabel("F1 when flagging the points with the largest reach")
ax.set_ylim(0, 1.05)
ax.legend()
ax.set_title("A local view (small h) is useless; following the chain to the end gives F1 = 1\n"
             "(connected components = h → ∞)")
fig.suptitle("04 · The colour is a long-range property: the information has to travel along the whole chain",
             fontweight="bold")
save(fig, "04_long_range.png")

# ---------------------------------------------------------------- 05 k matters
ks = range(1, 9)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
ax = axes[0]
for (ZZ, yy, name, col) in [(Z, y, "chains (walk)", "k"), (Zc, yc, "chains (circles)", "#2f6fdf")]:
    ax.plot(list(ks), [f1_top(yy, comp_size(ZZ, k)[0] + 1e-6 * np.random.default_rng(1).random(len(yy))) for k in ks],
            "o-", c=col, lw=2, label=name)
ax.axhline(2 * y.mean() / (1 + y.mean()), ls="--", c=GREY, label="all-red baseline")
ax.set_xlabel("k (neighbours per point in the graph)")
ax.set_ylabel("F1 of connected components")
ax.set_ylim(0, 1.05)
ax.legend()
ax.set_title("A chain point has exactly 2 neighbours:\nk = 2 fits, larger k links different chains")
i0 = np.where(y == 1)[0][0]
win = np.linalg.norm(X - X[i0], axis=1) < 160
for ax, k in zip(axes[1:], (2, 4)):
    A_k = knn_graph(Z, k)
    _, lab_k, nc_k = comp_size(Z, k)
    rr, cc = sparse.triu(A_k).nonzero()
    keep = win[rr] & win[cc]
    cross = lab[rr] != lab[cc]                                  # edge between two different true chains
    ax.add_collection(LineCollection(np.stack([X[rr[keep & ~cross]], X[cc[keep & ~cross]]], 1), colors=GREY, lw=1.2))
    ax.add_collection(LineCollection(np.stack([X[rr[keep & cross]], X[cc[keep & cross]]], 1), colors=GOLD, lw=2.2))
    ax.scatter(*X[win & (y == 0)].T, s=12, c=GREEN)
    ax.scatter(*X[win & (y == 1)].T, s=14, c=RED)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"k = {k}: {nc_k} components in total\n{(cross[keep]).sum()} edges between different chains (gold)")
fig.suptitle("05 · Choosing k: the graph must not add edges between chains", fontweight="bold")
save(fig, "05_choose_k.png")

# ---------------------------------------------------------------- 06 homophily across datasets
rows = []
for name, f in DATA.items():
    XX, ZZ, yy = load(f)
    A10 = knn_graph(ZZ, 10)
    rr, cc = A10.nonzero()
    red_to_red = yy[cc[yy[rr] == 1]].mean()
    A2 = knn_graph(ZZ, 2)
    r2, c2 = A2.nonzero()
    red_to_red2 = yy[c2[yy[r2] == 1]].mean()
    graph = f1_top(yy, comp_size(ZZ, 2)[0] + 1e-6 * np.random.default_rng(2).random(len(yy)))
    d10 = NearestNeighbors(n_neighbors=11).fit(ZZ).kneighbors(ZZ)[0][:, 10]
    local = max(f1_top(yy, d10), f1_top(yy, -d10))
    rows.append((name, red_to_red, red_to_red2, graph, local))
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
names = [r[0] for r in rows]
xx = np.arange(len(rows))
ax = axes[0]
ax.bar(xx - .2, [r[1] for r in rows], .4, color=GREY, label="k = 10 graph")
ax.bar(xx + .2, [r[2] for r in rows], .4, color="k", label="k = 2 graph")
ax.axhline(y.mean(), ls="--", c=RED, label="share of red in the data (no structure)")
ax.set_xticks(xx, names); ax.set_ylim(0, 1.05)
ax.set_ylabel("edges from a red point that lead to a red point")
ax.legend(fontsize=9)
ax.set_title("Homophily: do edges connect the same colour?")
ax = axes[1]
ax.bar(xx - .2, [r[3] for r in rows], .4, color="#2f6fdf", label="graph: connected components (k = 2)")
ax.bar(xx + .2, [r[4] for r in rows], .4, color=GREY, label="best of k-NN density / sparsity (k = 10)")
ax.axhline(2 * y.mean() / (1 + y.mean()), ls="--", c=GREY)
ax.set_xticks(xx, names); ax.set_ylim(0, 1.05)
ax.set_ylabel("F1 (no labels)")
ax.legend(fontsize=9)
ax.set_title("The graph wins only where its edges are homophilous\nAND the colour depends on long-range structure")
fig.suptitle("06 · Same method, four datasets: when does a graph help?", fontweight="bold")
save(fig, "06_when_graphs_help.png")

# ---------------------------------------------------------------- 07 all methods on the walk chains
n = y.sum()
cv = StratifiedKFold(5, shuffle=True, random_state=0)
d = {k: NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)[0][:, k] for k in (5, 10)}
S_ = knn_graph(Z, 10) + sparse.eye(len(y))
Dg = sparse.diags(np.asarray(S_.sum(1)).ravel() ** -.5)
S_ = Dg @ S_ @ Dg
km = KMeans(2, n_init=10, random_state=0).fit_predict(Z)
gm = GaussianMixture(2, n_init=5, random_state=0).fit(Z).predict_proba(Z)
sc = SpectralClustering(2, affinity="precomputed", random_state=0).fit_predict(knn_graph(Z, 10).toarray())
db = DBSCAN(eps=0.03, min_samples=2).fit_predict(Z)
big = np.bincount(db[db >= 0]).argmax()
E = TSNE(perplexity=30, init="pca", random_state=42).fit_transform(Z)
dE = NearestNeighbors(n_neighbors=11).fit(E).kneighbors(E)[0][:, 10]
idx = NearestNeighbors(n_neighbors=16).fit(Z).kneighbors(Z)[1][:, 1:]
F4 = StandardScaler().fit_transform(PolynomialFeatures(4, include_bias=False).fit_transform(Z))
lr = cross_val_predict(LogisticRegression(C=1e4, class_weight="balanced", max_iter=20000), F4, y, cv=cv,
                       method="predict_proba")[:, 1]
glp = np.zeros(len(y))
for tr, te in cv.split(Z, y):
    Y0 = np.zeros((len(y), 2)); Y0[tr, y[tr]] = 1; Y0[tr] /= np.array([[(y[tr] == 0).sum(), (y[tr] == 1).sum()]])
    Fp = Y0.copy()
    for _ in range(50):
        Fp = .9 * (S_ @ Fp) + .1 * Y0
    glp[te] = (Fp[:, 1] / (Fp.sum(1) + 1e-12))[te]
mino = lambda l: l if l.mean() < .5 else 1 - l
res = [
    ("Graph: connected components (k = 2)", f1_top(y, comp_size(Z, 2)[0] + 1e-6 * np.random.default_rng(3).random(len(y))), "graph"),
    ("k-NN density (k = 10)", f1_top(y, -d[10]), "unsup"),
    ("k-NN sparsity (k = 5)", f1_top(y, d[5]), "unsup"),
    ("LOF (k = 20)", f1_top(y, -LocalOutlierFactor(20).fit(Z).negative_outlier_factor_), "unsup"),
    ("K-Means", f1_score(y, mino(km)), "unsup"),
    ("GMM (2 components)", max(f1_top(y, gm[:, 0]), f1_top(y, gm[:, 1])), "unsup"),
    ("DBSCAN (outside largest cluster)", f1_score(y, db != big), "unsup"),
    ("t-SNE + k-NN density", f1_top(y, -dE), "unsup"),
    ("Graph: smoothed density (k = 10)", f1_top(y, S_ @ (-d[10])), "unsup"),
    ("Graph: spectral cut (k = 10)", f1_score(y, mino(sc)), "unsup"),
    ("k-NN classifier (uses labels)", f1_top(y, y[idx].mean(1) + 1e-6 * np.random.default_rng(4).random(len(y))), "sup"),
    ("Graph: label propagation (uses labels)", f1_top(y, glp), "sup"),
    ("Logistic regression, degree 4 (uses labels)", f1_top(y, lr), "sup"),
]
res.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(10, 6))
cols = {"graph": "#2f6fdf", "unsup": GREY, "sup": GOLD}
ax.barh([r[0] for r in res], [r[1] for r in res], color=[cols[r[2]] for r in res])
for i, r in enumerate(res):
    ax.text(r[1] + .01, i, f"{r[1]:.3f}", va="center", fontsize=9)
ax.axvline(2 * y.mean() / (1 + y.mean()), ls="--", c="k", lw=1)
ax.text(2 * y.mean() / (1 + y.mean()) + .01, -0.9, "all-red baseline", fontsize=8)
ax.set_xlim(0, 1.12)
ax.set_xlabel("F1 for red")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=cols["graph"], label="unsupervised graph method"),
                   Patch(color=cols["unsup"], label="other unsupervised"),
                   Patch(color=cols["sup"], label="supervised (uses labels)")], loc="lower right")
fig.suptitle("07 · Walk chains: the unsupervised graph method beats every other method, even supervised ones",
             fontweight="bold")
save(fig, "07_all_methods.png")
