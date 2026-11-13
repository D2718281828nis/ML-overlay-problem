"""Structural fuzzy clustering for the overlay problem (no labels used for fitting).

Classic Fuzzy C-Means works on raw (X, Y) and uses one fixed membership rule,
u_ij ~ d_ij^(-2/(m-1)). On these datasets the colour is not a matter of
position, so FCM only cuts the plane in two (F1 ~ 0.23, below the all-red
baseline). This script keeps the fuzzy idea but changes two things.

1. **What is clustered: structural / distribution features of each point**
   - ``sparse``: log distance to the 5th neighbour (local sparsity; original data)
   - ``density``: log distance to the 15th neighbour (wider density; ring data)
   - ``chain``: log size of the point's chain = connected component of the
     symmetric k=2 nearest-neighbour graph (long-range structure; chain data)
   - ``linear``: anisotropy of the 8-neighbourhood, 1 - lambda_min/lambda_max
     (is the point on a line-like structure?)
   - ``radial``: distance to the data centroid (optional, global shape)

2. **How membership is computed: a parametric membership function per cluster
   and per feature, whose parameters are estimated from that cluster's
   fuzzy-weighted distribution** and re-estimated every iteration:
   - ``gaussian``   exp(-(x-c)^2 / (2 (w*s)^2)), c, s = weighted mean / std
   - ``triangular`` asymmetric triangle: feet at weighted quantiles q(a), q(1-a)
                    widened by ``width``, peak at the weighted median
   - ``trapezoid``  plateau between q(0.5-p/2) and q(0.5+p/2) (p = ``plateau``),
                    feet as for the triangle
   - ``bell``       1 / (1 + |(x-c)/(w*s)|^(2b)), b = ``slope``
   - ``empirical``  the cluster's own fuzzy-weighted KDE of the feature, scaled to
                    peak 1 (bandwidth = ``width`` x Silverman), i.e. the membership
                    function *is* the observed distribution

   Feature memberships are aggregated with a t-norm (``product``, ``min``) or a
   weighted ``mean``. Each feature has a weight w_f that is learned without labels
   from how well the current fuzzy clusters separate on it (a fuzzy Fisher ratio
   raised to ``beta``; beta = 0 gives equal weights). The partition is then

       u_ij = (pi_j * mu_j(x_i)) ^ (1/(m-1)) / sum_k (pi_k * mu_k(x_i)) ^ (1/(m-1))

   so ``m`` keeps its FCM meaning (m -> 1 hard, large m -> uniform) and pi_j is
   the fuzzy cluster size (handles the 13% red / 87% green imbalance).

Red is the smaller fuzzy cluster (label-free). Labels are used only for scoring.

Usage
    python fuzzy_structural.py                                   # all datasets, defaults
    python fuzzy_structural.py --data df_red_green_overlay.csv --shape triangular --width 1.5
    python fuzzy_structural.py --sweep                           # shapes x widths x m table
    python fuzzy_structural.py --data df_red_green_chains.csv --plot fuzzy.png
"""

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

DATASETS = [
    "df_red_green_overlay.csv",
    "df_red_green_distinct.csv",
    "df_red_green_chains.csv",
    "df_red_green_circles.csv",
]
DEFAULT_FEATURES = ["sparse", "density", "chain", "linear"]
SHAPES = ["gaussian", "triangular", "trapezoid", "bell", "empirical"]
EPS = 1e-12


# ----------------------------------------------------------------------------
# Structural / distribution features
# ----------------------------------------------------------------------------
def structural_features(xy, names=DEFAULT_FEATURES, k_sparse=5, k_density=15,
                        k_chain=2, k_linear=8):
    """Return an (n, len(names)) matrix of label-free per-point features."""
    xy = np.asarray(xy, dtype=float)
    k_max = max(k_sparse, k_density, k_linear)
    dist, idx = NearestNeighbors(n_neighbors=k_max + 1).fit(xy).kneighbors(xy)
    cols = []
    for name in names:
        if name == "sparse":
            cols.append(np.log(dist[:, k_sparse] + EPS))
        elif name == "density":
            cols.append(np.log(dist[:, k_density] + EPS))
        elif name == "chain":
            a = kneighbors_graph(xy, k_chain, mode="connectivity", include_self=False)
            _, comp = connected_components(((a + a.T) > 0).astype(float), directed=False)
            cols.append(np.log(np.bincount(comp)[comp].astype(float)))
        elif name == "linear":
            nb = xy[idx[:, : k_linear + 1]]                     # self + k neighbours
            nb = nb - nb.mean(axis=1, keepdims=True)
            cov = np.einsum("nki,nkj->nij", nb, nb) / k_linear
            ev = np.linalg.eigvalsh(cov)                        # ascending
            cols.append(1.0 - ev[:, 0] / (ev[:, 1] + EPS))
        elif name == "radial":
            cols.append(np.linalg.norm(xy - xy.mean(axis=0), axis=1))
        else:
            raise ValueError(f"unknown feature {name!r}")
    f = np.column_stack(cols)
    # Robust standardisation so every feature lives on a comparable scale.
    med = np.median(f, axis=0)
    iqr = np.subtract(*np.percentile(f, [75, 25], axis=0))
    return (f - med) / np.where(iqr > EPS, iqr, f.std(axis=0) + EPS)


# ----------------------------------------------------------------------------
# Weighted statistics
# ----------------------------------------------------------------------------
def weighted_quantile(x, w, q):
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    cw = np.cumsum(ws) - 0.5 * ws
    cw /= ws.sum()
    return np.interp(q, cw, xs)


def weighted_mean_std(x, w):
    mu = np.sum(w * x) / w.sum()
    return mu, np.sqrt(np.sum(w * (x - mu) ** 2) / w.sum()) + EPS


# ----------------------------------------------------------------------------
# Membership functions (each fitted to one cluster's fuzzy-weighted feature)
# ----------------------------------------------------------------------------
class MembershipFunction:
    """Parametric membership mu(x) in [0, 1] whose parameters come from data.

    width   : spread multiplier (the main fuzziness knob; larger = fuzzier sets)
    alpha   : tail quantile used for the triangle/trapezoid feet
    plateau : quantile mass of the trapezoid's flat top
    slope   : exponent b of the generalised bell (larger = steeper sides)
    floor   : minimum membership, so a zero in one feature cannot veto a point
    min_spread : smallest allowed spread (features are scaled to IQR = 1), so a
              cluster sitting on one discrete value does not shrink to a spike
    """

    def __init__(self, shape="gaussian", width=1.0, alpha=0.05, plateau=0.5,
                 slope=2.0, floor=1e-3, min_spread=0.25):
        if shape not in SHAPES:
            raise ValueError(f"shape must be one of {SHAPES}")
        self.shape, self.width, self.alpha = shape, width, alpha
        self.plateau, self.slope, self.floor = plateau, slope, floor
        self.min_spread = min_spread

    def fit(self, x, w):
        self.c, self.s = weighted_mean_std(x, w)
        self.s = max(self.s, self.min_spread)       # resolution limit (discrete features)
        if self.shape in ("triangular", "trapezoid"):
            med = weighted_quantile(x, w, 0.5)
            lo = weighted_quantile(x, w, self.alpha)
            hi = weighted_quantile(x, w, 1 - self.alpha)
            # Feet widened by `width` around the median; asymmetric on purpose.
            self.a = med - self.width * max(med - lo, self.min_spread)
            self.d = med + self.width * max(hi - med, self.min_spread)
            if self.shape == "triangular":
                self.b = self.cc = med
            else:
                half = self.plateau / 2
                self.b = weighted_quantile(x, w, 0.5 - half)
                self.cc = weighted_quantile(x, w, 0.5 + half)
        elif self.shape == "empirical":
            self.grid = np.linspace(x.min() - 3 * self.s, x.max() + 3 * self.s, 512)
            n_eff = w.sum() ** 2 / np.sum(w ** 2)
            h = self.width * 1.06 * self.s * n_eff ** (-0.2)
            # Binned weighted KDE: weighted histogram on the grid, then Gaussian smoothing.
            step = self.grid[1] - self.grid[0]
            hist, _ = np.histogram(x, bins=len(self.grid), range=(self.grid[0] - step / 2,
                                                                  self.grid[-1] + step / 2), weights=w)
            dens = gaussian_filter1d(hist, h / step, mode="constant")
            self.curve = dens / max(dens.max(), EPS)
        return self

    def __call__(self, x):
        if self.shape == "gaussian":
            mu = np.exp(-0.5 * ((x - self.c) / (self.width * self.s)) ** 2)
        elif self.shape == "bell":
            mu = 1.0 / (1.0 + np.abs((x - self.c) / (self.width * self.s)) ** (2 * self.slope))
        elif self.shape in ("triangular", "trapezoid"):
            left = (x - self.a) / max(self.b - self.a, EPS)
            right = (self.d - x) / max(self.d - self.cc, EPS)
            mu = np.clip(np.minimum(np.minimum(left, right), 1.0), 0.0, 1.0)
        else:  # empirical
            mu = np.interp(x, self.grid, self.curve, left=0.0, right=0.0)
        return np.maximum(mu, self.floor)


# ----------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------
class StructuralFuzzyClustering:
    """Fuzzy clustering with per-cluster, per-feature learned membership functions."""

    def __init__(self, n_clusters=2, shape="triangular", width=1.5, m=2.0,
                 tnorm="product", beta=2.0, alpha=0.05, plateau=0.5, slope=2.0,
                 floor=1e-3, min_spread=0.25, use_prior=True, max_iter=200, tol=1e-5,
                 n_init=3, random_state=0):
        if m <= 1:
            raise ValueError("m must be > 1")
        if tnorm not in ("product", "min", "mean"):
            raise ValueError("tnorm must be product, min or mean")
        self.n_clusters, self.shape, self.width, self.m = n_clusters, shape, width, m
        self.tnorm, self.beta, self.use_prior = tnorm, beta, use_prior
        self.mf_kwargs = dict(shape=shape, width=width, alpha=alpha, plateau=plateau,
                              slope=slope, floor=floor, min_spread=min_spread)
        self.max_iter, self.tol, self.n_init = max_iter, tol, n_init
        self.random_state = random_state

    # -- helpers --------------------------------------------------------------
    def _feature_memberships(self, F):
        """mu[i, j, f]: membership of point i in cluster j on feature f."""
        n, p = F.shape
        mu = np.empty((n, self.n_clusters, p))
        for j in range(self.n_clusters):
            for f in range(p):
                mu[:, j, f] = self.mfs_[j][f](F[:, f])
        return mu

    def _aggregate(self, mu):
        w = self.feature_weights_[None, None, :]
        if self.tnorm == "product":
            return np.exp(np.sum(w * np.log(mu), axis=2))
        if self.tnorm == "min":
            return np.min(mu ** w, axis=2)
        return np.sum(w * mu, axis=2) / w.sum()

    def _partition(self, agg):
        prior = self.prior_[None, :] if self.use_prior else 1.0
        z = np.log(prior * agg + EPS) / (self.m - 1.0)
        z -= z.max(axis=1, keepdims=True)
        u = np.exp(z)
        return u / u.sum(axis=1, keepdims=True)

    def _m_step(self, F, u):
        wts = u ** self.m
        p = F.shape[1]
        self.mfs_ = [[MembershipFunction(**self.mf_kwargs).fit(F[:, f], wts[:, j])
                      for f in range(p)] for j in range(self.n_clusters)]
        self.prior_ = u.mean(axis=0)
        # Fuzzy Fisher ratio per feature: between-cluster spread / within-cluster spread.
        means = np.array([[weighted_mean_std(F[:, f], wts[:, j])[0] for f in range(p)]
                          for j in range(self.n_clusters)])
        stds = np.array([[weighted_mean_std(F[:, f], wts[:, j])[1] for f in range(p)]
                         for j in range(self.n_clusters)])
        sep = means.std(axis=0) / (stds.mean(axis=0) + EPS)
        w = (sep + EPS) ** self.beta
        self.feature_weights_ = p * w / w.sum()

    def _init_partition(self, F, rng):
        """FCM on the features (classic m=2) as a starting point."""
        u = rng.dirichlet(np.ones(self.n_clusters), size=len(F))
        for _ in range(100):
            wts = u ** 2
            c = (wts.T @ F) / wts.sum(axis=0)[:, None]
            d = np.linalg.norm(F[:, None, :] - c[None], axis=2) + EPS
            new = d ** -2.0
            new /= new.sum(axis=1, keepdims=True)
            if np.max(np.abs(new - u)) < 1e-6:
                break
            u = new
        return new

    @staticmethod
    def _split_feature(x):
        """Label-free 1-D two-group split of one feature (1-D 2-means on the values)."""
        lo, hi = np.percentile(x, [15, 85])
        for _ in range(50):
            thr = (lo + hi) / 2
            a, b = x[x <= thr], x[x > thr]
            if len(a) == 0 or len(b) == 0:
                break
            lo, hi = a.mean(), b.mean()
        z = (x - (lo + hi) / 2) / (np.std(x) * 0.25 + EPS)
        s = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - s, s])

    def _initial_partitions(self, F, rng):
        """Starts: one 1-D split per feature (so any single structural cue can seed
        the model) plus random-restart FCM on all features."""
        starts = [self._split_feature(F[:, f]) for f in range(F.shape[1])]
        starts += [self._init_partition(F, rng) for _ in range(self.n_init)]
        return starts

    def _objective(self, F, u):
        """Label-free criterion used only to pick among starts: the mixture
        log-likelihood after normalising every membership function to a density
        (equal feature weights, so different runs are comparable)."""
        n, p = F.shape
        logdens = np.zeros((n, self.n_clusters))
        for j in range(self.n_clusters):
            for f in range(p):
                grid = np.linspace(F[:, f].min() - 1, F[:, f].max() + 1, 2000)
                z = np.trapezoid(self.mfs_[j][f](grid), grid)
                logdens[:, j] += np.log(self.mfs_[j][f](F[:, f]) / z + EPS)
        logdens += np.log(self.prior_ + EPS)[None, :]
        mx = logdens.max(axis=1, keepdims=True)
        return float(np.sum(mx[:, 0] + np.log(np.exp(logdens - mx).sum(axis=1))))

    # -- API ------------------------------------------------------------------
    def fit(self, F):
        F = np.asarray(F, dtype=float)
        rng = np.random.default_rng(self.random_state)
        best = None
        self.runs_ = []
        for u in self._initial_partitions(F, rng):
            self._m_step(F, u)
            for it in range(1, self.max_iter + 1):
                new = self._partition(self._aggregate(self._feature_memberships(F)))
                change = np.max(np.abs(new - u))
                u = new
                self._m_step(F, u)
                if change < self.tol:
                    break
            score = self._objective(F, u)
            self.runs_.append((score, u[:, np.argmin(u.sum(axis=0))]))
            if best is None or score > best[0]:
                best = (score, u, self.mfs_, self.prior_, self.feature_weights_, it)
        _, self.u_, self.mfs_, self.prior_, self.feature_weights_, self.n_iter_ = best
        self.red_cluster_ = int(np.argmin(self.u_.sum(axis=0)))   # minority = red
        return self

    def soft_predict(self, F):
        return self._partition(self._aggregate(self._feature_memberships(np.asarray(F, float))))

    def red_membership(self, F=None):
        u = self.u_ if F is None else self.soft_predict(F)
        return u[:, self.red_cluster_]


# ----------------------------------------------------------------------------
# Classic FCM on (X, Y), for comparison (same as the notebook)
# ----------------------------------------------------------------------------
def classic_fcm_xy(xy, m=2.0, seed=42):
    x = (xy - xy.mean(axis=0)) / xy.std(axis=0)
    rng = np.random.default_rng(seed)
    u = rng.dirichlet(np.ones(2), size=len(x))
    for _ in range(300):
        w = u ** m
        c = (w.T @ x) / w.sum(axis=0)[:, None]
        d = np.linalg.norm(x[:, None] - c[None], axis=2) + EPS
        new = d ** (-2 / (m - 1))
        new /= new.sum(axis=1, keepdims=True)
        if np.max(np.abs(new - u)) < 1e-5:
            break
        u = new
    red = int(np.argmin(new.sum(axis=0)))
    return new[:, red]


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def load(path):
    df = pd.read_csv(path)
    return df[["X", "Y"]].to_numpy(float), df["Color"].str.contains("red").to_numpy().astype(int)


def score(y, red_mu):
    pred = (red_mu > 0.5).astype(int)
    return dict(f1=f1_score(y, pred, zero_division=0),
                prec=precision_score(y, pred, zero_division=0),
                rec=recall_score(y, pred, zero_division=0),
                ap=average_precision_score(y, red_mu),
                flagged=pred.mean())


def run(path, features, **kw):
    xy, y = load(path)
    F = structural_features(xy, features)
    model = StructuralFuzzyClustering(**kw).fit(F)
    return model, xy, y, F, score(y, model.red_membership())


def baseline_f1(y):
    p = y.mean()
    return 2 * p / (1 + p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", nargs="*", default=DATASETS)
    ap.add_argument("--features", nargs="*", default=DEFAULT_FEATURES)
    ap.add_argument("--shape", default="triangular", choices=SHAPES)
    ap.add_argument("--width", type=float, default=1.5, help="spread multiplier of the sets")
    ap.add_argument("--m", type=float, default=2.0, help="partition fuzziness (>1)")
    ap.add_argument("--tnorm", default="product", choices=["product", "min", "mean"])
    ap.add_argument("--beta", type=float, default=2.0, help="feature-weight sharpness (0 = equal)")
    ap.add_argument("--alpha", type=float, default=0.05, help="tail quantile for triangle feet")
    ap.add_argument("--plateau", type=float, default=0.5, help="trapezoid core mass")
    ap.add_argument("--slope", type=float, default=2.0, help="bell slope b")
    ap.add_argument("--floor", type=float, default=1e-3, help="minimum membership")
    ap.add_argument("--min-spread", type=float, default=0.25, help="smallest set spread (IQR units)")
    ap.add_argument("--no-prior", action="store_true", help="drop the cluster-size prior")
    ap.add_argument("--sweep", action="store_true", help="grid over shape x width x m")
    ap.add_argument("--plot", help="save a figure (first dataset) to this path")
    a = ap.parse_args()

    kw = dict(shape=a.shape, width=a.width, m=a.m, tnorm=a.tnorm, beta=a.beta,
              alpha=a.alpha, plateau=a.plateau, slope=a.slope, floor=a.floor,
              min_spread=a.min_spread,
              use_prior=not a.no_prior)

    if a.sweep:
        rows = []
        for shape, width, m in itertools.product(SHAPES, [0.7, 1.0, 1.5, 2.0], [1.5, 2.0]):
            row = dict(shape=shape, width=width, m=m)
            for path in a.data:
                row[path.replace("df_red_green_", "").replace(".csv", "")] = round(
                    run(path, a.features, **{**kw, "shape": shape, "width": width, "m": m})[4]["f1"], 3)
            rows.append(row)
            print(row, flush=True)
        print("\nF1 by setting (labels are used only to score, never to fit):")
        print(pd.DataFrame(rows).to_string(index=False))
        return

    print(f"features={a.features}  " + "  ".join(f"{k}={v}" for k, v in kw.items()))
    print(f"{'dataset':<28}{'baseline':>9}{'FCM(X,Y)':>10}{'struct F1':>10}{'prec':>7}{'rec':>7}"
          f"{'AP':>7}{'flag%':>7}  feature weights")
    for i, path in enumerate(a.data):
        model, xy, y, F, s = run(path, a.features, **kw)
        fcm = f1_score(y, (classic_fcm_xy(xy) > 0.5).astype(int))
        wts = " ".join(f"{n}={w:.2f}" for n, w in zip(a.features, model.feature_weights_))
        print(f"{path:<28}{baseline_f1(y):>9.3f}{fcm:>10.3f}{s['f1']:>10.3f}{s['prec']:>7.3f}"
              f"{s['rec']:>7.3f}{s['ap']:>7.3f}{100 * s['flagged']:>6.1f}%  {wts}")
        if a.plot and i == 0:
            plot(model, xy, y, F, a.features, path, a.plot)


def plot(model, xy, y, F, names, path, out):
    import matplotlib.pyplot as plt
    p = len(names)
    fig, axes = plt.subplots(1, p + 1, figsize=(4.2 * (p + 1), 4))
    red = model.red_membership()
    sc = axes[0].scatter(xy[:, 0], xy[:, 1], c=red, cmap="RdYlGn_r", s=6, vmin=0, vmax=1)
    axes[0].scatter(*xy[y == 1].T, facecolors="none", edgecolors="k", s=14, lw=0.4, label="true red")
    axes[0].set_title(f"red membership\n{path}")
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right", fontsize=7)
    fig.colorbar(sc, ax=axes[0])
    for f in range(p):
        ax = axes[f + 1]
        grid = np.linspace(F[:, f].min(), F[:, f].max(), 400)
        ax.hist(F[y == 0, f], bins=40, density=True, alpha=0.3, color="green", label="green")
        ax.hist(F[y == 1, f], bins=40, density=True, alpha=0.3, color="red", label="red")
        twin = ax.twinx()
        for j, col in zip(range(model.n_clusters), ["tab:blue", "tab:orange"]):
            lab = "red set" if j == model.red_cluster_ else "green set"
            twin.plot(grid, model.mfs_[j][f](grid), color=col, lw=2, label=lab)
        twin.set_ylim(0, 1.05)
        ax.set_title(f"{names[f]} (weight {model.feature_weights_[f]:.2f})")
        if f == 0:
            ax.legend(loc="upper left", fontsize=7)
            twin.legend(loc="upper right", fontsize=7)
    fig.suptitle(f"Structural fuzzy clustering: shape={model.shape}, width={model.width}, m={model.m}")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"figure saved to {out}")


if __name__ == "__main__":
    main()
