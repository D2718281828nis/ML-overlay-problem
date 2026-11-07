"""Hyperbolic GCN for red/green node classification (NumPy, no PyTorch needed).

Re-implements `HyperbolicGCN` from ML-MachineLearning-Graphs
(healthcare-gnn-lab/src/healthcare_gnn/models/gnn.py) with the Poincaré-ball
operations of models/hyperbolic.py (curvature -c):

    tangent  = x W1 + b1
    ball     = expmap0(tangent)                     # lift into the Poincaré ball
    agg      = S logmap0(ball) W2 + b2              # GCN aggregation in the tangent space
    ball_out = project(expmap0(relu(agg)))
    logit    = logmap0(ball_out) w3 + b3

S = D~^(-1/2)(A + I)D~^(-1/2) is the GCN operator of a symmetric k-NN graph.
Training: class-balanced logistic loss + Adam; gradients are written by hand and
were checked against finite differences (max error ~1e-7).

Note: logmap0(expmap0(v)) = v except where the ball's numerical edge clips v, so
in this architecture the curvature acts as a bound on the hidden-vector length:
c -> 0 is a plain Euclidean GCN.
"""
import numpy as np
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

EPS = 1e-5


def project(x, c):
    n = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)
    return x * np.minimum(1, (1 - EPS) / np.sqrt(c) / n)


def expmap0(v, c):
    n = np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-7)
    r = np.sqrt(c)
    return project(np.tanh(r * n) * v / (r * n), c)


def logmap0(x, c):
    x = project(x, c)
    n = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-7)
    r = np.sqrt(c)
    return np.arctanh(np.minimum(r * n, 1 - 1e-7)) * x / (r * n)


def _lift(v, c):
    return logmap0(expmap0(v, c), c)


def _lift_grad(v, g, c):
    """Exact derivative of the lift, which is the radial clip v * min(1, M/|v|)."""
    out = _lift(v, c)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    m = np.linalg.norm(out, axis=1, keepdims=True)
    clipped = (n > m * (1 + 1e-9)).ravel()
    G = g.copy()
    if clipped.any():
        u = v[clipped] / n[clipped]
        G[clipped] = (m[clipped] / n[clipped]) * (g[clipped] - u * np.sum(g[clipped] * u, 1, keepdims=True))
    return G


def gcn_operator(Z, k):
    A = kneighbors_graph(Z, k, mode="connectivity", include_self=False)
    A = ((A + A.T) > 0).astype(float) + sparse.eye(len(Z))
    d = np.asarray(A.sum(1)).ravel() ** -0.5
    return (sparse.diags(d) @ A @ sparse.diags(d)).tocsr()


def forward(P, X, S, c):
    T = X @ P["W1"] + P["b1"]
    L1 = _lift(T, c)
    A = S @ L1 @ P["W2"] + P["b2"]
    R = np.maximum(A, 0)
    L2 = _lift(R, c)
    return L2 @ P["w3"] + P["b3"], (T, L1, A, R, L2)


def _grads(P, X, S, c, y, w, idx):
    z, (T, L1, A, R, L2) = forward(P, X, S, c)
    p = 1 / (1 + np.exp(-z))
    dz = np.zeros_like(z)
    dz[idx] = (w[idx] * (p[idx] - y[idx])) / w[idx].sum()
    g = {"w3": L2.T @ dz, "b3": dz.sum(0)}
    dA = _lift_grad(R, dz @ P["w3"].T, c) * (A > 0)
    g["W2"] = (S @ L1).T @ dA
    g["b2"] = dA.sum(0)
    dT = _lift_grad(T, S.T @ (dA @ P["W2"].T), c)
    g["W1"] = X.T @ dT
    g["b1"] = dT.sum(0)
    return g


def train(X, S, y, idx, c=1.0, hidden=16, epochs=300, lr=0.02, seed=0):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    P = {"W1": rng.normal(0, 1 / np.sqrt(d), (d, hidden)), "b1": np.zeros(hidden),
         "W2": rng.normal(0, 1 / np.sqrt(hidden), (hidden, hidden)), "b2": np.zeros(hidden),
         "w3": rng.normal(0, 1 / np.sqrt(hidden), (hidden, 1)), "b3": np.zeros(1)}
    yy = y.reshape(-1, 1).astype(float)
    n1 = y[idx].sum()
    w = np.where(yy == 1, len(idx) / (2 * n1), len(idx) / (2 * (len(idx) - n1)))   # class balance
    m1 = {k: np.zeros_like(v) for k, v in P.items()}
    m2 = {k: np.zeros_like(v) for k, v in P.items()}
    for t in range(1, epochs + 1):
        g = _grads(P, X, S, c, yy, w, idx)
        for k in P:
            m1[k] = .9 * m1[k] + .1 * g[k]
            m2[k] = .999 * m2[k] + .001 * g[k] ** 2
            P[k] -= lr * (m1[k] / (1 - .9 ** t)) / (np.sqrt(m2[k] / (1 - .999 ** t)) + 1e-8)
    return P


def node_features(Z):
    """Standardised X, Y, their squares and product, and the log 10-NN distance."""
    d10 = NearestNeighbors(n_neighbors=11).fit(Z).kneighbors(Z)[0][:, 10]
    return StandardScaler().fit_transform(np.c_[PolynomialFeatures(2, include_bias=False).fit_transform(Z), np.log(d10)])


def hgcn_cv_scores(Z, y, k=10, c=1.0, folds=5, seed=0, **kw):
    """Out-of-fold red scores: each node is scored by a model that never saw its colour."""
    y = np.asarray(y).astype(int)
    X, S = node_features(Z), gcn_operator(Z, k)
    out = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(Z, y):
        P = train(X, S, y, tr, c=c, seed=seed, **kw)
        out[te] = forward(P, X, S, c)[0].ravel()[te]
    return out
