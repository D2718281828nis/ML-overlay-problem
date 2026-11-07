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

Chain modes (connectivity example, where an unsupervised GRAPH method wins):
  --red chains     --red-chains long chains holding --n-red points in total
  --green chains   many short chains of --chain-len MIN MAX points (total --n-green)
  Every chain is a smooth random walk (--chain-shape walk) or a closed circle
  (--chain-shape circle, radius = points * step / 2pi) with a fixed --step (12 px) inside F;
  points of different chains keep at least --gap (30 px) apart. Both colours then have the same
  local density and the same local shape (every point is a piece of a chain); only the
  LENGTH of the connected chain differs. Example:
      python make_distinct_dataset.py --green chains --red chains --out df_red_green_chains.csv
      python make_distinct_dataset.py --green chains --red chains --chain-shape circle --out df_red_green_circles.csv

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
ap.add_argument("--green", choices=["uniform", "gaussian", "chains"], default="uniform", help="green distribution")
ap.add_argument("--n-green", type=float, default=1445, help="green count (Poisson mean for uniform, exact for gaussian)")
ap.add_argument("--green-sigma", type=float, default=400.0, help="std of the green Gaussian, px")
ap.add_argument("--green-center", type=float, nargs=2, metavar=("X", "Y"), default=None,
                help="centre of the green Gaussian, px (default: footprint centre)")
ap.add_argument("--red", choices=["ring", "gaussian", "uniform", "chains"], default="ring", help="red distribution")
ap.add_argument("--n-red", type=int, default=227, help="red count")
ap.add_argument("--center", type=float, nargs=2, metavar=("X", "Y"), default=None,
                help="centre of the red ring / Gaussian, px (default: original point nearest the centroid, (1046, 904))")
ap.add_argument("--radius", type=float, default=350.0, help="ring radius R, px")
ap.add_argument("--sigma", type=float, default=15.0, help="ring width (std across the circle) or red Gaussian std, px")
ap.add_argument("--clip-red", action="store_true",
                help="also keep red inside the green footprint (redraw red points that fall outside)")
ap.add_argument("--red-chains", type=int, default=6, help="chains mode: number of long red chains")
ap.add_argument("--chain-len", type=int, nargs=2, metavar=("MIN", "MAX"), default=(12, 20), help="chains mode: green chain length range")
ap.add_argument("--step", type=float, default=12.0, help="chains mode: distance between consecutive points of a chain, px")
ap.add_argument("--gap", type=float, default=30.0, help="chains mode: minimum distance between different chains, px")
ap.add_argument("--turn", type=float, default=0.35, help="chains mode: std of the turning angle per step, rad")
ap.add_argument("--chain-shape", choices=["walk", "circle"], default="walk",
                help="chains mode: open smooth random walks or closed circles")
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


class ChainField:
    """Grows self-avoiding chains (smooth random walks with a fixed step) inside F.
    A candidate point is rejected if it is closer than `gap` to any point of another
    chain, or to an older part (> 3 steps back) of its own chain."""

    def __init__(self, step, gap, turn):
        self.step, self.gap, self.turn = step, gap, turn
        self.pts, self.owner, self.grid = [], [], {}

    def _key(self, p):
        return int(p[0] // self.gap), int(p[1] // self.gap)

    def _blocked(self, p, chain, own_start, own_len):
        a, b = self._key(p)
        for i in range(a - 1, a + 2):
            for j in range(b - 1, b + 2):
                for q in self.grid.get((i, j), ()):
                    if self.owner[q] == chain and q >= own_start + own_len - 3:
                        continue                      # own recent points are allowed
                    if np.hypot(*(self.pts[q] - p)) < self.gap:
                        return True
        return False

    def _add(self, p, chain):
        self.pts.append(p); self.owner.append(chain)
        self.grid.setdefault(self._key(p), []).append(len(self.pts) - 1)

    def _undo(self, start):
        del self.pts[start:]; del self.owner[start:]
        self.grid = {}
        for i, p in enumerate(self.pts):
            self.grid.setdefault(self._key(p), []).append(i)

    def circle(self, n, chain):
        """Place one closed circle of n points (step = chord between neighbours); returns n or 0."""
        r = self.step / (2 * np.sin(np.pi / n))
        phi = 2 * np.pi * np.arange(n) / n
        for _ in range(600):
            c = P[rng.integers(len(P))] + rng.normal(0, 20, 2)
            pts = c + r * np.c_[np.cos(phi), np.sin(phi)] @ self._rot(rng.uniform(0, 2 * np.pi))
            if inside(pts).all() and not any(self._blocked(q, chain, len(self.pts), 0) for q in pts):
                for q in pts:
                    self._add(q, chain)
                return n
        return 0

    @staticmethod
    def _rot(a):
        return np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])

    def grow(self, n, chain):
        """Try to grow one chain of n points; returns how many points it got."""
        if args.chain_shape == "circle":
            return self.circle(n, chain)
        start = len(self.pts)
        for _ in range(200):
            p = P[rng.integers(len(P))] + rng.normal(0, 20, 2)
            if inside(p[None])[0] and not self._blocked(p, chain, start, 0):
                break
        else:
            return 0
        self._add(p, chain); theta = rng.uniform(0, 2 * np.pi)
        while len(self.pts) - start < n:
            for _ in range(30):
                t = theta + rng.normal(0, self.turn)
                q = self.pts[-1] + self.step * np.array([np.cos(t), np.sin(t)])
                if inside(q[None])[0] and not self._blocked(q, chain, start, len(self.pts) - start):
                    break
            else:
                break
            theta = t; self._add(q, chain)
        return len(self.pts) - start

    def chains(self, total, lengths, chain_id0, exact):
        """Grow chains until `total` points; lengths(i) gives the wanted length of chain i.
        exact=True retries a chain until it reaches full length (used for long red chains)."""
        got, cid, fails, streak = 0, chain_id0, 0, 0
        while got < total and fails < 20000 and streak < 300:   # give up when the area is full
            want = min(lengths(cid - chain_id0), total - got)
            start = len(self.pts); n = self.grow(want, cid)
            if n < want if exact else n < min(args.chain_len[0], want):
                self._undo(start); fails += 1; streak += 1; continue
            got += n; cid += 1; streak = 0
        if got < total:
            print(f"warning: the footprint is full, placed {got} of {total} points "
                  f"(try fewer/shorter chains, a smaller --gap or another --seed)")
        return np.array([p for p, o in zip(self.pts, self.owner) if chain_id0 <= o < cid])


field = ChainField(args.step, args.gap, args.turn)
if args.red == "chains":             # long red chains first, they need the room
    k = args.red_chains
    red = field.chains(args.n_red, lambda i: args.n_red // k + (1 if i < args.n_red % k else 0), 1_000_000, exact=True)

if args.green == "uniform":
    green = draw_inside(uniform, rng.poisson(args.n_green))
elif args.green == "gaussian":
    green = draw_inside(blob(green_center, args.green_sigma), int(round(args.n_green)))
else:
    lo_len, hi_len = args.chain_len
    green = field.chains(int(round(args.n_green)), lambda i: int(rng.integers(lo_len, hi_len + 1)), 0, exact=False)

if args.red != "chains":
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
