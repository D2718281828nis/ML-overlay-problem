"""Generate a variant of df_red_green_overlay.csv where red and green follow
clearly different spatial distributions, while keeping the original footprint.

* green: homogeneous Poisson process (= uniform) inside the original point cloud's
  footprint. Its count is drawn as Poisson(1445), so it averages the original count.
* red:   2-D Gaussian (isotropic) centred inside the footprint, 227 points,
  clipped to the footprint.

The footprint is the set of locations within FOOTPRINT_R pixels of any original
point, so the overall placement (shape, holes, extent) is preserved.

Usage:  python make_distinct_dataset.py   ->  writes df_red_green_distinct.csv
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SEED = 42
FOOTPRINT_R = 45.0      # px: how far from an original point still counts as "inside"
RED_SIGMA = 90.0        # px: std of the red Gaussian
N_RED = 227             # same as the original dataset
GREEN_RATE = 1445       # Poisson mean for the green count (original count)

rng = np.random.default_rng(SEED)
orig = pd.read_csv("df_red_green_overlay.csv")
P = orig[["X", "Y"]].to_numpy(float)
tree = cKDTree(P)
inside = lambda pts: tree.query(pts)[0] <= FOOTPRINT_R

lo, hi = P.min(0) - FOOTPRINT_R, P.max(0) + FOOTPRINT_R


def uniform_in_footprint(n):
    out = []
    while sum(len(o) for o in out) < n:
        c = rng.uniform(lo, hi, size=(4 * n, 2))
        out.append(c[inside(c)])
    return np.vstack(out)[:n]


# Red centre: the original point closest to the centroid (guaranteed inside).
centre = P[tree.query(P.mean(0))[1]]


def gaussian_in_footprint(n):
    out = []
    while sum(len(o) for o in out) < n:
        c = rng.normal(centre, RED_SIGMA, size=(4 * n, 2))
        out.append(c[inside(c)])
    return np.vstack(out)[:n]


green = uniform_in_footprint(rng.poisson(GREEN_RATE))
red = gaussian_in_footprint(N_RED)

df = pd.DataFrame(
    np.vstack([green, red]).round().astype(int), columns=["X", "Y"]
)
df["Color"] = ["light_green"] * len(green) + ["red"] * len(red)
df["Type"] = ["cross"] * len(green) + ["point"] * len(red)
df.to_csv("df_red_green_distinct.csv", index=False)

print(f"green={len(green)}  red={len(red)}  red centre={centre}  sigma={RED_SIGMA}")
