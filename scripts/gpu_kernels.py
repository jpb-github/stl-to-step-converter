#!/usr/bin/env python
"""Fast ray kernels for the STL skill: local thickness and hidden-line visibility.

One primitive, `first_hit`, casts rays against a triangle soup and returns the distance to the nearest
hit (inf if none). Everything else is built on it:

  local_thickness(tri)                 ray from each triangle centroid along its inward normal
  occluded(P, view_dir, tri, bias)     True where a point on the part is hidden from a viewer in direction view_dir

Backends (backend="auto" picks the first available):
  numba  CPU, all cores (prange over rays), float64, any-hit early exit. Default.
  cuda   torch on a CUDA GPU, float32, chunked Moller-Trumbore. Opt-in: about as fast as numba, but float32
         can miss or invent a grazing hit (2 of 73k thickness rays differed on the test part)
  numpy  CPU fallback, vectorised in chunks, slow

tri is an (N,3,3) float array of triangles (as returned by load_stl). Rays and triangles are in mm.
A hit needs t > tmin (default 1e-4 mm), so a ray leaving a surface does not hit its own triangle.

Usage as a script (self test / timing):
  python gpu_kernels.py PART.stl [--n 4000] [--backend auto|cuda|numba|numpy]
"""
import argparse, time
import numpy as np


def load_stl(path):
    d = open(path, "rb").read()
    n = int(np.frombuffer(d[80:84], "<u4")[0])
    if 84 + 50 * n == len(d):
        a = np.frombuffer(d[84:84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
        return a["v"].reshape(-1, 3, 3).astype(float)
    import re
    v = np.array([[float(x) for x in m] for m in re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", d.decode("ascii", "ignore"))])
    return v.reshape(-1, 3, 3)


def available_backends():
    """Order = what backend="auto" picks. numba (float64, all cores) beats CUDA on this workload: same speed
    for thickness, faster for occlusion (early exit), and exact. CUDA stays available with backend="cuda"."""
    out = []
    try:
        import numba  # noqa: F401
        out.append("numba")
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available(): out.append("cuda")
    except Exception:
        pass
    out.append("numpy")
    return out


def _pick(backend):
    av = available_backends()
    if backend == "auto": return av[0]
    if backend not in av: raise RuntimeError(f"backend {backend} not available; have {av}")
    return backend


# ---------------------------------------------------------------- numpy
def _first_hit_numpy(O, D, V0, E1, E2, tmin, skip, chunk=64):
    out = np.full(len(O), np.inf)
    for s in range(0, len(O), chunk):
        o, d = O[s:s + chunk, None], D[s:s + chunk, None]
        p = np.cross(d, E2[None]); det = (E1[None] * p).sum(-1)
        ok = np.abs(det) > 1e-12
        inv = 1.0 / np.where(ok, det, 1.0)
        tv = o - V0[None]
        u = (tv * p).sum(-1) * inv
        q = np.cross(tv, E1[None])
        v = (d * q).sum(-1) * inv
        t = (E2[None] * q).sum(-1) * inv
        hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > tmin)
        if skip is not None:
            hit[np.arange(len(hit)), skip[s:s + chunk]] = False
        out[s:s + chunk] = np.where(hit, t, np.inf).min(1)
    return out


# ---------------------------------------------------------------- numba
_nb_kernel = None


def _get_numba_kernel():
    global _nb_kernel
    if _nb_kernel is not None: return _nb_kernel
    from numba import njit, prange

    @njit(parallel=True, fastmath=True, cache=True)
    def kern(O, D, V0, E1, E2, tmin, skip, any_hit, out):
        for i in prange(O.shape[0]):
            ox, oy, oz = O[i, 0], O[i, 1], O[i, 2]
            dx, dy, dz = D[i, 0], D[i, 1], D[i, 2]
            best = np.inf
            for j in range(V0.shape[0]):
                if j == skip[i]: continue
                e2x, e2y, e2z = E2[j, 0], E2[j, 1], E2[j, 2]
                px = dy * e2z - dz * e2y; py = dz * e2x - dx * e2z; pz = dx * e2y - dy * e2x
                e1x, e1y, e1z = E1[j, 0], E1[j, 1], E1[j, 2]
                det = e1x * px + e1y * py + e1z * pz
                if abs(det) < 1e-12: continue
                inv = 1.0 / det
                tx, ty, tz = ox - V0[j, 0], oy - V0[j, 1], oz - V0[j, 2]
                u = (tx * px + ty * py + tz * pz) * inv
                if u < -1e-9 or u > 1.0 + 1e-9: continue
                qx = ty * e1z - tz * e1y; qy = tz * e1x - tx * e1z; qz = tx * e1y - ty * e1x
                v = (dx * qx + dy * qy + dz * qz) * inv
                if v < -1e-9 or u + v > 1.0 + 1e-9: continue
                t = (e2x * qx + e2y * qy + e2z * qz) * inv
                if t > tmin and t < best:
                    best = t
                    if any_hit: break
            out[i] = best
    _nb_kernel = kern
    return kern


# ---------------------------------------------------------------- cuda
def _first_hit_cuda(O, D, V0, E1, E2, tmin, skip, mem_gb=2.0):
    import torch
    dev = "cuda"
    f = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
    # keep float32 numbers small: work relative to the mesh centre
    c = V0.mean(0)
    V0t, E1t, E2t = f(V0 - c), f(E1), f(E2)
    Ot, Dt = f(O - c), f(D)
    sk = torch.as_tensor(skip, device=dev) if skip is not None else None
    T = V0t.shape[0]
    C = max(16, int(mem_gb * 1e9 / (T * 4 * 12)))      # about 12 live (C,T) float32 tensors
    out = torch.empty(len(O), device=dev)
    inf = torch.tensor(float("inf"), device=dev)
    idx_t = torch.arange(T, device=dev)
    eps = 1e-4          # float32 edge tolerance: a ray through a shared edge must not slip between two triangles
    for s in range(0, len(O), C):
        o, d = Ot[s:s + C, None], Dt[s:s + C, None]
        p = torch.cross(d.expand(-1, T, -1), E2t[None].expand(d.shape[0], -1, -1), dim=-1)
        det = (E1t[None] * p).sum(-1)
        ok = det.abs() > 1e-9
        inv = 1.0 / torch.where(ok, det, torch.ones_like(det))
        tv = o - V0t[None]
        u = (tv * p).sum(-1) * inv
        q = torch.cross(tv, E1t[None].expand(tv.shape[0], -1, -1), dim=-1)
        v = (d * q).sum(-1) * inv
        t = (E2t[None] * q).sum(-1) * inv
        hit = ok & (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps) & (t > tmin)
        if sk is not None:
            hit &= idx_t[None] != sk[s:s + C, None]
        out[s:s + C] = torch.where(hit, t, inf).min(1).values
    return out.cpu().numpy().astype(float)


# ---------------------------------------------------------------- public API
def first_hit(O, D, tri, tmin=1e-4, skip=None, any_hit=False, backend="auto"):
    """Distance along each ray (origin O[i], unit direction D[i]) to the nearest triangle hit, inf if none.
    skip[i] = index of a triangle that ray i must ignore (its own). any_hit only speeds up the numba backend;
    the result is then the distance to *a* hit, not necessarily the nearest."""
    O = np.ascontiguousarray(O, float); D = np.ascontiguousarray(D, float)
    tri = np.asarray(tri, float)
    V0, E1, E2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    skip_a = np.full(len(O), -1, np.int64) if skip is None else np.asarray(skip, np.int64)
    b = _pick(backend)
    if b == "cuda":
        return _first_hit_cuda(O, D, V0, E1, E2, tmin, skip_a)
    if b == "numba":
        out = np.empty(len(O))
        _get_numba_kernel()(O, D, np.ascontiguousarray(V0), np.ascontiguousarray(E1), np.ascontiguousarray(E2),
                            float(tmin), skip_a, bool(any_hit), out)
        return out
    return _first_hit_numpy(O, D, V0, E1, E2, tmin, skip_a)


def triangle_normals(tri):
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    L = np.linalg.norm(n, axis=1)
    return n / np.where(L > 0, L, 1)[:, None], L / 2


def local_thickness(tri, backend="auto", sel=None):
    """Distance from each triangle centroid to the first surface hit along its inward normal (-normal; STL
    normals point outward). inf = no hit; zero-area triangles also give inf. sel = optional index subset."""
    tri = np.asarray(tri, float)
    n, area = triangle_normals(tri)
    idx = np.arange(len(tri)) if sel is None else np.asarray(sel)
    ok = area[idx] > 0
    out = np.full(len(idx), np.inf)
    if ok.any():
        i = idx[ok]
        out[ok] = first_hit(tri[i].mean(1), -n[i], tri, skip=i, backend=backend)
    return out


def occluded(P, view_dir, tri, bias=0.02, backend="auto"):
    """True where the point P[i] (on the part surface) is hidden from a viewer looking along -view_dir
    (view_dir points from the part toward the viewer). bias (mm) ignores the surface the point sits on."""
    P = np.asarray(P, float)
    d = np.asarray(view_dir, float); d = d / np.linalg.norm(d)
    D = np.broadcast_to(d, P.shape).copy()
    return np.isfinite(first_hit(P, D, tri, tmin=bias, any_hit=True, backend=backend))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stl"); ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--backend", default="auto")
    a = ap.parse_args()
    tri = load_stl(a.stl)
    print(f"{len(tri)} triangles; backends {available_backends()}")
    sel = np.random.default_rng(0).choice(len(tri), min(a.n, len(tri)), replace=False)
    t0 = time.time(); th = local_thickness(tri, a.backend, sel); dt = time.time() - t0
    print(f"{_pick(a.backend)}: {len(sel)} rays in {dt:.2f} s; full mesh would take about {dt * len(tri) / len(sel):.1f} s")
    f = th[np.isfinite(th)]
    print(f"hits {len(f)}/{len(th)}, median thickness {np.median(f):.3f} mm, min {f.min():.3f}")
