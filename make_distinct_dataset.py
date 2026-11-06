"""Generate a variant of df_red_green_overlay.csv where red and green follow
clearly different spatial distributions, while keeping the original footprint.

* green: homogeneous Poisson process (= uniform density) inside the original
  point cloud's footprint F. Count N_g ~ Poisson(1445), the original count.
      p_green(x) = 1 / |F|                                  for x in F
* red:   a noisy ring ("circle as the baseline"). 227 points with
      theta ~ Uniform(0, 2*pi),   r ~ Normal(RING_R, RING_SIGMA^2)
      x = centre + r * (cos theta, sin theta)
  i.e. a Gaussian profile across the circle of radius RING_R; points that fall
  outside F are re-drawn.
      p_red(x) ~ N(|x - centre|; RING_R, RING_SIGMA^2) / (2*pi*|x - centre|)

The footprint F is the set of locations within FOOTPRINT_R pixels of any original
point, so the overall placement (shape, holes, extent) is preserved.

Usage:  python make_distinct_dataset.py   ->  writes df_red_green_distinct.csv
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SEED = 42
FOOTPRINT_R = 45.0      # px: how far from an original point still counts as "inside"
RING_R = 350.0          # px: radius of the red circle (95% of it lies inside F)
RING_SIGMA = 15.0       # px: std of the Gaussian profile across the circle
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


# Ring centre: the original point closest to the centroid.
centre = P[tree.query(P.mean(0))[1]]


def ring_in_footprint(n):
    out = []
    while sum(len(o) for o in out) < n:
        theta = rng.uniform(0, 2 * np.pi, size=4 * n)
        r = rng.normal(RING_R, RING_SIGMA, size=4 * n)
        c = centre + r[:, None] * np.c_[np.cos(theta), np.sin(theta)]
        out.append(c[inside(c)])
    return np.vstack(out)[:n]


green = uniform_in_footprint(rng.poisson(GREEN_RATE))
red = ring_in_footprint(N_RED)

df = pd.DataFrame(
    np.vstack([green, red]).round().astype(int), columns=["X", "Y"]
)
df["Color"] = ["light_green"] * len(green) + ["red"] * len(red)
df["Type"] = ["cross"] * len(green) + ["point"] * len(red)
df.to_csv("df_red_green_distinct.csv", index=False)

print(f"green={len(green)}  red={len(red)}  ring centre={centre}  R={RING_R}  sigma={RING_SIGMA}")
