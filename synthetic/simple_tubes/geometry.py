"""Tube-tree geometry for simple_v1 (plan §3.1–3.4, truncation from §3.7).

Coordinates are continuous (z, y, x) in voxel units. Voxel centres sit at integer
coordinates 0..n-1 on each axis, so the volume covers [-0.5, n - 0.5] per axis, with
shape = (nz, ny, nx).

One neuron = a tree of branches. Each branch is a persistent random walk of control
points, interpolated with a centripetal Catmull-Rom spline (or a smoothing cubic
B-spline), resampled at a fixed arc-length step, cut at its first exit from the
volume and extended straight out past the boundary. Children start exactly on a
resampled point of their parent, so every neuron is one connected piece.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GeometryParams:
    shape: tuple = (400, 600, 600)         # (Z, Y, X)
    step_len: tuple = (12.0, 20.0)          # control-point spacing, U(lo, hi)
    turn_sigma_deg: float = 20.0            # turn per step ~ |N(0, sigma)| ...
    turn_max_deg: float = 45.0              # ... capped here
    root_points: tuple = (10, 30)           # control points per root, inclusive
    face_start_prob: float = 0.5            # root starts on a face (else inside)
    spline: str = "catmull-rom"             # or "bspline"
    bspline_smoothing: float = 4.0          # splprep s per control point (bspline only)
    dense_step: float = 0.05                # spline evaluation step before resampling
    resample_step: float = 0.25             # arc-length spacing of the centreline
    child_lambda: tuple = (2.5, 1.5, 0.8, 0.3)  # Poisson mean by parent order; len = max order
    branch_t: tuple = (0.15, 0.85)          # attach point, fraction of parent arc length
    branch_min_sep: float = 10.0            # arc length between attach points on a parent
    branch_angle_deg: tuple = (30.0, 75.0)  # child direction vs parent tangent
    branch_tries: int = 20                  # attempts to place one child
    child_points_frac: tuple = (0.3, 0.7)   # child control points / parent control points
    child_radius_frac: tuple = (0.7, 0.85)  # child start radius / parent radius at attach
    root_radius: tuple = (1.5, 3.0)
    taper: tuple = (0.1, 0.3)               # linear radius loss along a branch
    radius_floor: float = 1.5               # PPP prep opening (radius 1) erodes thinner tubes
    deep_margin: float = 1.0                # "inside" for truncation/attach = this deep
    # simple_v2 additions; the defaults reproduce simple_v1 exactly (no extra random draws)
    root_radius_log: bool = False           # draw the root radius log-uniformly in root_radius
    branchiness: tuple = None               # per-neuron factor on child_lambda, log-uniform (lo, hi)


@dataclass
class Branch:
    branch_id: int
    parent_id: int            # -1 for the root
    order: int
    t_parent: float           # arc-length fraction on the parent's core (nan for the root)
    control: np.ndarray       # (K, 3) control points
    points: np.ndarray        # (M, 3) resampled centreline: core + exit extension
    radius: np.ndarray        # (M,)
    n_core: int               # points[:n_core] follow the spline; the rest is the exit extension
    truncated: bool
    r_start: float
    taper: float

    def to_json(self) -> dict:
        return {
            "branch_id": self.branch_id,
            "parent_id": self.parent_id,
            "order": self.order,
            "t_parent": None if math.isnan(self.t_parent) else round(self.t_parent, 4),
            "control_points": np.round(self.control, 3).tolist(),
            "r_start": round(self.r_start, 4),
            "taper": round(self.taper, 4),
            "n_points": int(len(self.points)),
            "n_core": int(self.n_core),
            "truncated": bool(self.truncated),
        }


@dataclass
class Neuron:
    start_kind: str                       # "face" or "interior"
    branches: list = field(default_factory=list)
    branchiness: float = 1.0              # factor on child_lambda (simple_v2)

    def to_json(self) -> dict:
        return {"start_kind": self.start_kind, "branchiness": round(self.branchiness, 4),
                "branches": [b.to_json() for b in self.branches]}


# ---------------------------------------------------------------- vectors

def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def random_unit(rng: np.random.Generator) -> np.ndarray:
    return _unit(rng.normal(size=3))


def random_perpendicular(d: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    while True:
        v = rng.normal(size=3)
        v -= v.dot(d) * d
        n = np.linalg.norm(v)
        if n > 1e-6:
            return v / n


def rotate_away(d: np.ndarray, angle: float, rng: np.random.Generator) -> np.ndarray:
    """Rotate unit vector d by `angle` (rad) about a random axis perpendicular to d."""
    u = random_perpendicular(d, rng)
    return math.cos(angle) * d + math.sin(angle) * u


# ---------------------------------------------------------------- curves

def random_walk(p0, d0, n_points, P: GeometryParams, rng, turn_first=True) -> np.ndarray:
    """Persistent random walk (plan §3.1). Returns (n_points, 3) control points."""
    sigma = math.radians(P.turn_sigma_deg)
    cap = math.radians(P.turn_max_deg)
    pts = [np.asarray(p0, float)]
    d = _unit(np.asarray(d0, float))
    for k in range(n_points - 1):
        if k > 0 or turn_first:
            d = rotate_away(d, min(abs(rng.normal(0.0, sigma)), cap), rng)
        pts.append(pts[-1] + rng.uniform(*P.step_len) * d)
    return np.array(pts)


def catmull_rom(control: np.ndarray, dense_step=0.05, alpha=0.5) -> np.ndarray:
    """Centripetal Catmull-Rom through every control point (Barry-Goldman form).

    End tangents come from reflected phantom points (2*P0 - P1, 2*Pn - Pn-1).
    Duplicated end points would give a zero knot interval and divide by zero.
    """
    P = np.asarray(control, float)
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = [P[:1]]
    for i in range(len(P) - 1):
        p0, p1, p2, p3 = ext[i:i + 4]
        t0 = 0.0
        t1 = t0 + np.linalg.norm(p1 - p0) ** alpha
        t2 = t1 + np.linalg.norm(p2 - p1) ** alpha
        t3 = t2 + np.linalg.norm(p3 - p2) ** alpha
        n = max(2, int(math.ceil(np.linalg.norm(p2 - p1) / dense_step)) + 1)
        t = np.linspace(t1, t2, n)[1:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.vstack(out)


def bspline(control: np.ndarray, smoothing_per_point=4.0, dense_step=0.05) -> np.ndarray:
    """Smoothing cubic B-spline. It does not pass through the control points, so the
    first control point is prepended to keep branch attachment exact."""
    from scipy.interpolate import splev, splprep

    P = np.asarray(control, float)
    k = min(3, len(P) - 1)
    tck, _ = splprep(P.T, k=k, s=smoothing_per_point * len(P))
    chord = np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
    u = np.linspace(0.0, 1.0, max(2, int(math.ceil(chord / dense_step)) + 1))
    curve = np.array(splev(u, tck)).T
    return np.vstack([P[:1], curve])


def resample_arclength(curve: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-9])
    curve = curve[keep]
    s = np.concatenate([[0.0], np.cumsum(seg[seg > 1e-9])])
    total = s[-1]
    targets = np.arange(int(math.floor(total / step)) + 1) * step
    if total - targets[-1] > 1e-6:
        targets = np.append(targets, total)
    return np.stack([np.interp(targets, s, curve[:, j]) for j in range(3)], axis=1)


def taper_radius(n_points: int, r_start: float, taper: float, floor: float) -> np.ndarray:
    frac = np.linspace(0.0, 1.0, n_points) if n_points > 1 else np.zeros(1)
    return np.maximum(floor, r_start * (1.0 - taper * frac))


def truncate_at_exit(points, radius, shape, step, deep_margin):
    """Cut a centreline at its first exit from the volume (plan §3.7).

    Leading points before the branch first gets `deep_margin` inside are kept from
    the last outside point on (a face start enters cleanly). After that, the first
    point outside ends the core, and a straight extension of length r + 1 continues
    along the exit direction. A ray that has left a box cannot re-enter it.
    Returns (points, radius, n_core, truncated) or None if the branch is never inside.
    """
    lo, hi = -0.5, np.asarray(shape, float) - 0.5
    inside = np.all((points >= lo) & (points <= hi), axis=1)
    deep = np.all((points >= lo + deep_margin) & (points <= hi - deep_margin), axis=1)
    if not deep.any():
        return None
    first_deep = int(np.argmax(deep))
    outside_before = np.nonzero(~inside[:first_deep])[0]
    begin = int(outside_before[-1]) if len(outside_before) else 0
    outside_after = np.nonzero(~inside[first_deep:])[0]
    if len(outside_after) == 0:
        return points[begin:], radius[begin:], len(points) - begin, False
    cut = first_deep + int(outside_after[0])
    d = _unit(points[cut] - points[cut - 1])
    r = float(radius[cut - 1])
    n_ext = int(math.ceil((r + 1.0) / step))
    ext = points[cut - 1] + d * (step * np.arange(1, n_ext + 1))[:, None]
    pts = np.vstack([points[begin:cut], ext])
    rad = np.concatenate([radius[begin:cut], np.full(n_ext, r)])
    return pts, rad, cut - begin, True


# ---------------------------------------------------------------- trees

def _make_branch(control, r_start, P: GeometryParams, rng, **meta):
    if P.spline == "catmull-rom":
        curve = catmull_rom(control, P.dense_step)
    elif P.spline == "bspline":
        curve = bspline(control, P.bspline_smoothing, P.dense_step)
    else:
        raise ValueError(f"unknown spline {P.spline!r}")
    pts = resample_arclength(curve, P.resample_step)
    taper = float(rng.uniform(*P.taper))
    rad = taper_radius(len(pts), r_start, taper, P.radius_floor)
    cut = truncate_at_exit(pts, rad, P.shape, P.resample_step, P.deep_margin)
    if cut is None:
        return None
    pts, rad, n_core, truncated = cut
    return Branch(control=control, points=pts, radius=rad, n_core=n_core,
                  truncated=truncated, r_start=float(r_start), taper=taper, **meta)


def _pick_attach(parent: Branch, used: list, P: GeometryParams, rng):
    """Index on the parent's core for a child start, or None."""
    n = parent.n_core
    i_lo = int(math.ceil(P.branch_t[0] * (n - 1)))
    i_hi = int(math.floor(P.branch_t[1] * (n - 1)))
    if i_hi <= i_lo or i_lo < 1:
        return None
    lo = -0.5 + P.deep_margin
    hi = np.asarray(P.shape, float) - 0.5 - P.deep_margin
    for _ in range(P.branch_tries):
        i = int(rng.integers(i_lo, i_hi + 1))
        p = parent.points[i]
        if not np.all((p >= lo) & (p <= hi)):
            continue
        if any(abs(i - j) * P.resample_step < P.branch_min_sep for j in used):
            continue
        return i
    return None


def build_neuron(rng: np.random.Generator, P: GeometryParams) -> Neuron | None:
    """One branching neuron. None if the root never gets inside the volume."""
    S = np.asarray(P.shape, float)
    if rng.random() < P.face_start_prob:
        axis = int(rng.integers(3))
        side = int(rng.integers(2))
        p0 = rng.uniform(0, S - 1)
        p0[axis] = -0.5 if side == 0 else S[axis] - 0.5
        d0 = np.zeros(3)
        d0[axis] = 1.0 if side == 0 else -1.0
        kind = "face"
    else:
        p0 = rng.uniform(0, S - 1)
        d0 = random_unit(rng)
        kind = "interior"
    k_root = int(rng.integers(P.root_points[0], P.root_points[1] + 1))
    control = random_walk(p0, d0, k_root, P, rng, turn_first=True)
    if P.root_radius_log:
        lo, hi = P.root_radius
        r0 = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    else:
        r0 = float(rng.uniform(*P.root_radius))
    root = _make_branch(control, r0, P, rng, branch_id=0, parent_id=-1, order=0,
                        t_parent=float("nan"))
    if root is None:
        return None
    factor = 1.0
    if P.branchiness:
        lo, hi = P.branchiness
        factor = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    neuron = Neuron(start_kind=kind, branches=[root], branchiness=factor)
    queue = [root]
    while queue:
        parent = queue.pop(0)
        if parent.order >= len(P.child_lambda):
            continue
        n_children = int(rng.poisson(P.child_lambda[parent.order] * factor))
        used: list = []
        for _ in range(n_children):
            i = _pick_attach(parent, used, P, rng)
            if i is None:
                continue
            used.append(i)
            tangent = _unit(parent.points[i + 1] - parent.points[i - 1])
            angle = math.radians(rng.uniform(*P.branch_angle_deg))
            d0 = rotate_away(tangent, angle, rng)
            k = max(2, int(round(rng.uniform(*P.child_points_frac) * len(parent.control))))
            control = random_walk(parent.points[i], d0, k, P, rng, turn_first=False)
            r_child = max(P.radius_floor,
                          float(rng.uniform(*P.child_radius_frac)) * float(parent.radius[i]))
            child = _make_branch(
                control, r_child, P, rng, branch_id=len(neuron.branches),
                parent_id=parent.branch_id, order=parent.order + 1,
                t_parent=i / (parent.n_core - 1))
            if child is None:   # cannot happen: the start is deep inside
                continue
            neuron.branches.append(child)
            queue.append(child)
    return neuron
