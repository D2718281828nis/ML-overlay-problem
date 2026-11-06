"""Generate a red/green dataset with chosen spatial distributions, keeping the
footprint of df_red_green_overlay.csv.

Footprint F = all locations within FOOTPRINT_R px of any original point (keeps the
overall placement, including holes). Green points always live inside F (draws that
land outside are redrawn). Red points follow their own distribution wherever it
falls, also outside F, unless --clip-red is given (then they are redrawn too).

Green distributions (--green):
  uniform   homogeneous Poisson process: N_g ~ Poisson(n_green), x ~ Uniform(F)
            p_g(x) = 1 / |F|
  gaussian  isotropic Gaussian blob around --green-center, std --green-sigma
            p_g(x) ~ N2(x; c_g, sigma_g^2 I)

Red distributions (--red):
  ring      noisy circle: theta ~ U(0, 2pi), r ~ N(radius, sigma^2),
            x = center + r (cos theta, sin theta)
            p_r(x) ~ N(|x - c|; R, sigma^2) / (2 pi |x - c|)
  gaussian  isotropic Gaussian blob around --center, std --sigma
  uniform   x ~ Uniform(F)  (same as green: no signal at all)

The defaults reproduce the committed df_red_green_distinct.csv exactly
(uniform green, unclipped red ring R=350, sigma=15, centre (1046, 904), seed 42).

Examples:
  python make_distinct_dataset.py
  python make_distinct_dataset.py --red gaussian --sigma 120 --center 800 700
  python make_distinct_dataset.py --green gaussian --green-sigma 400 --red ring \
      --radius 250 --sigma 25 --center 1200 1000 --out my_data.csv
Then point the scripts at it:  OVERLAY_DATA=my_data.csv python overlay_clustering_advanced.py
"""
import argparse

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

FOOTPRINT_R = 45.0  # px: how far from an original point still counts as "inside"

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--green", choices=["uniform", "gaussian"], default="uniform", help="green distribution")
ap.add_argument("--n-green", type=float, default=1445, help="green count (Poisson mean for uniform, exact for gaussian)")
ap.add_argument("--green-sigma", type=float, default=400.0, help="std of the green Gaussian, px")
ap.add_argument("--green-center", type=float, nargs=2, metavar=("X", "Y"), default=None,
                help="centre of the green Gaussian, px (default: footprint centre)")
ap.add_argument("--red", choices=["ring", "gaussian", "uniform"], default="ring", help="red distribution")
ap.add_argument("--n-red", type=int, default=227, help="red count")
ap.add_argument("--center", type=float, nargs=2, metavar=("X", "Y"), default=None,
                help="centre of the red ring / Gaussian, px (default: original point nearest the centroid, (1046, 904))")
ap.add_argument("--radius", type=float, default=350.0, help="ring radius R, px")
ap.add_argument("--sigma", type=float, default=15.0, help="ring width (std across the circle) or red Gaussian std, px")
ap.add_argument("--clip-red", action="store_true",
                help="also keep red inside the green footprint (redraw red points that fall outside)")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--source", default="df_red_green_overlay.csv", help="CSV whose footprint is kept")
ap.add_argument("--out", default="df_red_green_distinct.csv")
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
P = pd.read_csv(args.source)[["X", "Y"]].to_numpy(float)
tree = cKDTree(P)
inside = lambda pts: tree.query(pts)[0] <= FOOTPRINT_R
lo, hi = P.min(0) - FOOTPRINT_R, P.max(0) + FOOTPRINT_R
default_center = P[tree.query(P.mean(0))[1]]
center = np.array(args.center) if args.center else default_center
green_center = np.array(args.green_center) if args.green_center else default_center


def draw_inside(sampler, n):
    """Rejection sampling: keep drawing batches until n points fall inside F."""
    out = []
    while sum(len(o) for o in out) < n:
        c = sampler(4 * n)
        out.append(c[inside(c)])
    return np.vstack(out)[:n]


uniform = lambda m: rng.uniform(lo, hi, size=(m, 2))


def blob(c, s):
    return lambda m: rng.normal(c, s, size=(m, 2))


def ring(c, R, s):
    def sample(m):
        theta = rng.uniform(0, 2 * np.pi, size=m)
        r = rng.normal(R, s, size=m)
        return c + r[:, None] * np.c_[np.cos(theta), np.sin(theta)]
    return sample


if args.green == "uniform":
    green = draw_inside(uniform, rng.poisson(args.n_green))
else:
    green = draw_inside(blob(green_center, args.green_sigma), int(round(args.n_green)))

red_sampler = {"ring": ring(center, args.radius, args.sigma),
               "gaussian": blob(center, args.sigma),
               "uniform": uniform}[args.red]
# uniform red means Uniform(F), so it always stays inside the footprint
red = draw_inside(red_sampler, args.n_red) if (args.clip_red or args.red == "uniform") else red_sampler(args.n_red)

df = pd.DataFrame(np.vstack([green, red]).round().astype(int), columns=["X", "Y"])
df["Color"] = ["light_green"] * len(green) + ["red"] * len(red)
df["Type"] = ["cross"] * len(green) + ["point"] * len(red)
df.to_csv(args.out, index=False)

print(f"green={len(green)} ({args.green})  red={len(red)} ({args.red})  centre={center}  "
      f"R={args.radius}  sigma={args.sigma}  clip_red={args.clip_red}  seed={args.seed}  -> {args.out}")
print(f"red points outside the green footprint: {int((~inside(red)).sum())}")
