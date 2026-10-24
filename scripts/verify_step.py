#!/usr/bin/env python
"""Size test: compare a STEP model with its source STL by section outlines.

Usage:
  python verify_step.py PART.stl PART.step [--tol 0.05] [--per-axis 8] [--planes "x=15.9,y=222,z=40"]
                        [--supports supports.json] [--out OUTDIR] [--origin bbox-min|none] [--jobs N]

Both sections (STL and tessellated STEP) are cut on the same planes, each outline is sampled every
--step mm, and nearest distances are measured both ways with a KD-tree. Samples that fall inside a
support region (--supports) are ignored. A plane fails when the largest remaining distance exceeds --tol.
A failing plane is re-cut at a small offset: a grazing artefact (plane on a wall, chamfer end or cylinder
axis) disappears at the offset, a real error does not. Plane passes if either cut passes.

supports.json: list of regions to ignore (part coordinates after the origin shift):
  {"type":"box","min":[x,y,z],"max":[x,y,z]}
  {"type":"cylinders","axis":"z","centres":[[a,b],...],"r":2.3,"range":[-1,0.25]}   # a,b = the two other axes in XYZ order

Exit code: 0 all planes pass, 1 any plane fails, 2 input/model error.
Requires: numpy, scipy, cadquery. Optional: matplotlib (overlay PNGs of failing planes).
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial import cKDTree


def load_stl(path):
    d = open(path, "rb").read()
    n = int(np.frombuffer(d[80:84], "<u4")[0])
    if 84 + 50 * n == len(d):
        a = np.frombuffer(d[84:84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
        return a["v"].reshape(-1, 3, 3).astype(float)
    import re
    txt = d.decode("ascii", "ignore")
    v = np.array([[float(x) for x in m] for m in re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", txt)])
    return v.reshape(-1, 3, 3)


def load_step(path):
    import cadquery as cq
    shape = cq.importers.importStep(path)
    solids = shape.solids().vals()
    v = shape.val()
    info = {"valid": v.isValid(), "solids": len(solids), "faces": len(shape.faces().vals()), "volume": v.Volume()}
    return shape, info


def tessellate(shape, tol=0.002, ang=0.05):
    v, t = shape.val().tessellate(tol, ang)
    V = np.array([[p.x, p.y, p.z] for p in v])
    return V[np.array(t)]


def _tess_part(args):
    """Worker: import the STEP and tessellate every n-th face (k, n). Returns (M,3,3) triangles."""
    path, k, n, tol, ang = args
    import cadquery as cq
    out = []
    for f in cq.importers.importStep(path).faces().vals()[k::n]:
        v, t = f.tessellate(tol, ang)
        if len(t): out.append(np.array([[p.x, p.y, p.z] for p in v])[np.array(t)])
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


def load_and_tessellate(path, jobs, tol=0.002, ang=0.05):
    """Same result as load_step + tessellate, with the faces split over `jobs` processes.
    The main process loads the STEP for validity/volume while the workers tessellate."""
    if jobs <= 1:
        shape, info = load_step(path)
        return info, tessellate(shape, tol, ang)
    with ProcessPoolExecutor(jobs) as ex:
        futs = [ex.submit(_tess_part, (path, k, jobs, tol, ang)) for k in range(jobs)]
        shape, info = load_step(path)
        return info, np.concatenate([f.result() for f in futs])


def section(tri, axis, val):
    """Segments (N,2,2) of the plane axis=val through triangles (M,3,3), in the two other axes."""
    o = [i for i in range(3) if i != axis]
    d = tri[:, :, axis] - (val + 1e-7)
    cross = ~((d > 0).all(1) | (d < 0).all(1))
    T, D = tri[cross], d[cross]
    pts, ok = [], []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        c = D[:, i] * D[:, j] < 0
        f = np.where(c, D[:, i] / np.where(c, D[:, i] - D[:, j], 1), 0)
        pts.append((T[:, i] + f[:, None] * (T[:, j] - T[:, i]))[:, o]); ok.append(c)
    pts = np.stack(pts, 1); ok = np.stack(ok, 1)               # (n,3,2), (n,3)
    keep = ok.sum(1) == 2
    pts, ok = pts[keep], ok[keep]
    order = np.argsort(~ok, axis=1, kind="stable")[:, :2]
    a = np.take_along_axis(pts, order[:, :1, None], 1)[:, 0]
    b = np.take_along_axis(pts, order[:, 1:2, None], 1)[:, 0]
    return np.stack([a, b], 1)


def sample(segs, h):
    if len(segs) == 0: return np.zeros((0, 2))
    L = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
    n = np.maximum(1, np.ceil(L / h).astype(int))
    idx = np.repeat(np.arange(len(segs)), n)
    k = np.concatenate([(np.arange(m) + 0.5) / m for m in n])
    return segs[idx, 0] + (segs[idx, 1] - segs[idx, 0]) * k[:, None]


def to3(axis, val, P2):
    o = [i for i in range(3) if i != axis]
    Q = np.zeros((len(P2), 3)); Q[:, o] = P2; Q[:, axis] = val
    return Q


def support_mask(Q, regions):
    m = np.zeros(len(Q), bool)
    for r in regions:
        if r["type"] == "box":
            lo, hi = np.array(r["min"]), np.array(r["max"])
            m |= ((Q >= lo) & (Q <= hi)).all(1)
        elif r["type"] == "cylinders":
            ax = "xyz".index(r["axis"]); o = [i for i in range(3) if i != ax]
            C = np.array(r["centres"])
            inr = (Q[:, ax] >= r["range"][0]) & (Q[:, ax] <= r["range"][1])
            if inr.any():
                dd = np.linalg.norm(Q[inr][:, None, o] - C[None], axis=2).min(1)
                idx = np.where(inr)[0]; m[idx[dd < r["r"]]] = True
    return m


def compare(stl, stp, axis, val, h, regions):
    a = section(stp, axis, val); b = section(stl, axis, val)
    if len(a) == 0 or len(b) == 0:
        return None
    pa, pb = sample(a, h), sample(b, h)
    da = cKDTree(pb).query(pa)[0]; db = cKDTree(pa).query(pb)[0]
    qa, qb = to3(axis, val, pa), to3(axis, val, pb)
    ma, mb = support_mask(qa, regions), support_mask(qb, regions)
    return dict(da=da, db=db, qa=qa, qb=qb, ma=ma, mb=mb, sa=a, sb=b)


def stats(r):
    dA, dB = r["da"][~r["ma"]], r["db"][~r["mb"]]
    return (dA.max() if len(dA) else 0.0, dB.max() if len(dB) else 0.0,
            np.percentile(np.r_[dA, dB], 99) if len(dA) + len(dB) else 0.0)


_G = {}


def _init(stl, stp, regions, tol, h, offset):
    _G.update(stl=stl, stp=stp, regions=regions, tol=tol, h=h, offset=offset)


def _plane(task):
    """One plane, with the offset re-cut for grazing artefacts. Runs in a worker process.
    Returns (axis, plane used, stats, ok, worst-point text, comparison data for overlays if it failed) or None."""
    ax, val = task
    stl, stp, regions, tol, h, offset = (_G[k] for k in ("stl", "stp", "regions", "tol", "h", "offset"))
    r = compare(stl, stp, ax, val, h, regions)
    if r is None: return None
    s1 = stats(r); worst = max(s1[0], s1[1]); used = val; rr = r
    if worst > tol:
        r2 = compare(stl, stp, ax, val + offset, h, regions)
        if r2 is not None:
            s2 = stats(r2)
            if max(s2[0], s2[1]) < worst: worst, used, rr, s1 = max(s2[0], s2[1]), val + offset, r2, s2
    ok = worst <= tol
    loc = ""
    if not ok:
        cand = [(rr["da"][i], rr["qa"][i], "STEP") for i in [int(np.argmax(np.where(rr["ma"], -1, rr["da"])))]] + \
               [(rr["db"][i], rr["qb"][i], "STL") for i in [int(np.argmax(np.where(rr["mb"], -1, rr["db"])))]]
        d, q, who = max(cand, key=lambda c: c[0]); loc = f"worst {d:.2f} at ({q[0]:.1f}, {q[1]:.1f}, {q[2]:.1f}) on {who} outline"
    return ax, used, s1, ok, loc, (rr if not ok else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stl"); ap.add_argument("step")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--per-axis", type=int, default=8)
    ap.add_argument("--planes", default="")
    ap.add_argument("--supports", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--step-mm", type=float, default=0.02, dest="h")
    ap.add_argument("--offset", type=float, default=0.31)
    ap.add_argument("--origin", default="bbox-min", choices=["bbox-min", "none"])
    ap.add_argument("--jobs", type=int, default=0, help="worker processes for tessellation and planes (default min(6, cpu count); 1 = serial)")
    a = ap.parse_args()
    jobs = a.jobs or min(6, os.cpu_count() or 1)
    try:
        stl = load_stl(a.stl)
        info, stp = load_and_tessellate(a.step, jobs)
    except Exception as e:
        print("ERROR loading inputs:", e); return 2
    if a.origin == "bbox-min":
        stl = stl - stl.reshape(-1, 3).min(0)
    regions = json.load(open(a.supports)) if a.supports else []
    lo, hi = stl.reshape(-1, 3).min(0), stl.reshape(-1, 3).max(0)
    slo, shi = stp.reshape(-1, 3).min(0), stp.reshape(-1, 3).max(0)
    print(f"STEP: valid={info['valid']} solids={info['solids']} faces={info['faces']} volume={info['volume']:.1f}")
    print("bbox STL ", np.round(lo, 3), np.round(hi, 3)); print("bbox STEP", np.round(slo, 3), np.round(shi, 3))
    bbox_ok = np.abs(np.r_[slo - lo, shi - hi]).max()
    print(f"bbox max difference {bbox_ok:.3f}  (tol {a.tol})")
    fr = np.array([0.063, 0.171, 0.287, 0.402, 0.519, 0.634, 0.751, 0.869])
    fr = fr[:a.per_axis]
    planes = [(ax, lo[ax] + f * (hi[ax] - lo[ax])) for ax in range(3) for f in fr]
    for item in filter(None, a.planes.split(",")):
        k, v = item.split("="); planes.append(("xyz".index(k.strip().lower()), float(v)))
    fails = []; rows = []
    ctx =(stl, stp, regions, a.tol, a.h, a.offset)
    if jobs > 1 and len(planes) > 1:
        with ProcessPoolExecutor(min(jobs, len(planes)), initializer=_init, initargs=ctx) as ex:
            results = list(ex.map(_plane, planes))
    else:
        _init(*ctx); results = [_plane(p) for p in planes]
    for res in results:
        if res is None: continue
        ax, used, s1, ok, loc, rr = res
        rows.append((ax, used, s1[0], s1[1], s1[2], ok, loc))
        if not ok: fails.append((ax, used, rr))
    print(f"\n{'plane':<10}{'STEP->STL':>11}{'STL->STEP':>11}{'p99':>8}  result")
    for ax, used, d1, d2, p99, ok, loc in rows:
        print(f"{'XYZ'[ax]}={used:<8.2f}{d1:>11.3f}{d2:>11.3f}{p99:>8.3f}  {'ok' if ok else 'FAIL  ' + loc}")
    worst_all = max((max(r[2], r[3]) for r in rows), default=0)
    print(f"\n{len(rows)} planes, {len(fails)} fail, worst deviation {worst_all:.3f} mm (tol {a.tol})")
    if fails and a.out:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            from matplotlib.collections import LineCollection
            os.makedirs(a.out, exist_ok=True)
            for ax, used, r in fails:
                fig, axp = plt.subplots(figsize=(11, 8))
                axp.add_collection(LineCollection(r["sb"], colors="b", lw=0.8)); axp.add_collection(LineCollection(r["sa"], colors="r", lw=0.5))
                bad = np.r_[r["qa"][(r["da"] > a.tol) & ~r["ma"]], r["qb"][(r["db"] > a.tol) & ~r["mb"]]]
                o = [i for i in range(3) if i != ax]
                if len(bad): axp.plot(bad[:, o[0]], bad[:, o[1]], "g.", ms=2)
                axp.autoscale(); axp.set_aspect("equal"); axp.set_title(f"{'XYZ'[ax]}={used:.2f}: blue STL, red STEP, green = over tol")
                fig.savefig(os.path.join(a.out, f"fail_{'xyz'[ax]}_{used:.2f}.png"), dpi=70); plt.close(fig)
            print("overlays written to", a.out)
        except Exception as e:
            print("overlay skipped:", e)
    if not info["valid"] or info["solids"] != 1: print("STEP is not a single valid solid"); return 1
    return 1 if (fails or bbox_ok > a.tol) else 0


if __name__ == "__main__":
    sys.exit(main())
