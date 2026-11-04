"""A moving barrier that turns: NHTSA test 10072, the research oblique moving deformable barrier (OMDB, 2,437 kg)
at 90.33 km/h into a standing 2017 Nissan Altima (1,718 kg), 15°, 35% overlap, measured from the barrier's overhead
high-speed camera (No. 12 "Overhead of OMDB View", 28 mm lens, 1,000 fps).

  floor   a plane fitted to MoGe-3's points of the floor (frame at −50 ms)
  scale   the photo targets on the barrier's honeycomb face, 450 mm apart (NHTSA data sheet 8), at −50 ms before
          anything touches it (an evenly spaced column): their spacing on the plane at their own MoGe-3 height
  lens    from straight floor lines only if the lines agree (spread of the middle half under 0.1); they don't here
          (−0.40 to +0.40 line by line), so no correction; the constant-speed physics fit is reported beside it
  body    the barrier's front truss (its orange steel behind the honeycomb, rigid), a rigid body sliding and turning
          on the floor (hybrid.track_rigid); seeded at impact, followed back through the approach and on. Heights:
          per feature, or one plane at the truss's median, whichever turns the barrier less over the approach (it
          runs straight on its guide until impact)
  CG      980 mm behind the front axle, 597–605 mm behind the faceplate, 605 mm of honeycomb in front of that
          (data sheet 8): 2.185 m behind the honeycomb's front face, on the barrier's centre line
  check   approach against the speed trap; the barrier's acceleration in its own (turning) axes, integrated,
          against its CG accelerometers (curves 267 X, 268 Y; recorded to 194 ms)
Frame k of the video is k − 51 ms (frame 0 repeats −50). Usage: PYTHONPATH=backend backend/.venv/bin/python benchmark/omdb10072.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "benchmark"))
from physlens.hybrid import fit_distortion, lens_from_lines, plane_point, radial, track_rigid   # noqa: E402
from physlens.moving_camera import _plane_ransac   # noqa: E402
import nhtsa_sensors   # noqa: E402

CACHE = ROOT / "records" / "real_cache" / "omdb10072"
T0 = -50                                       # ms at frames_u8[0]
TRUTH = {"speed_kmh": 90.33, "m_barrier": 2437.0, "m_car": 1717.8, "target_spacing_m": 0.450,
         "cg_behind_face_m": 0.605 + 0.601 + 0.980, "source": "NHTSA test 10072, data sheet 8"}


def moge(k):
    z = np.load(CACHE / f"moge_{k}.npz")
    D, M, K = z["d"], z["m"] & np.isfinite(z["d"]), z["K"]
    h, w = D.shape
    ys, xs = np.mgrid[0:h, 0:w]
    return np.dstack([(xs - K[0, 2]) * D / K[0, 0], (ys - K[1, 2]) * D / K[1, 1], D]), M, K


def honeycomb(img, cart):
    """The white honeycomb block: bright, unsaturated, the largest such region touching the orange cart."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] > 200) & (hsv[..., 1] < 40)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    nl, lab, st, _ = cv2.connectedComponentsWithStats(m)
    near = cv2.dilate(cart.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    touch = [b for b in range(1, nl) if (near & (lab == b)).any()]
    return lab == max(touch, key=lambda b: st[b, cv2.CC_STAT_AREA]) if touch else np.zeros(cart.shape, bool)


def quartered_discs(img, min_r=10, max_r=30, min_score=40):
    """Photo targets (discs in four quadrants): circles whose ring alternates dark/light twice around."""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    cs = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=30, param1=120, param2=22, minRadius=min_r, maxRadius=max_r)
    ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    out = []
    for x, y, r in ([] if cs is None else cs[0]):
        ring = np.mean([g[(y + f * r * np.sin(ang)).astype(int).clip(0, g.shape[0] - 1),
                          (x + f * r * np.cos(ang)).astype(int).clip(0, g.shape[1] - 1)].astype(float)
                        for f in (0.4, 0.6, 0.8)], axis=0)
        sp = np.abs(np.fft.rfft(ring - ring.mean()))
        if sp[2] / (sp[1:].sum() + 1e-9) * np.ptp(ring) > min_score:
            out.append((float(x), float(y)))
    return np.array(out)


def main():
    U = np.load(CACHE / "frames_u8.npy")
    gray = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in U]
    h, w = gray[0].shape
    ys, xs = np.mgrid[0:h, 0:w]
    t = (np.arange(len(U)) + T0) / 1000.0
    k0 = -T0                                                             # impact (0 ms)

    P, M, K = moge(0)
    m0 = np.load(CACHE / "masks_0.npz")
    hc0 = honeycomb(U[0], m0["cart"])
    moving0 = cv2.dilate((m0["car"] | m0["cart"] | hc0).astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    moving0[:, int(np.argwhere(hc0)[:, 1].min()) - 20:] = True          # the barrier's side of the picture
    Q = P[M & ~moving0]
    n, c, _ = _plane_ransac(Q[::7], 0.01 * float(np.median(Q[:, 2])))
    if n[2] > 0:
        n = -n
    h0 = float(-(c @ n))
    e1 = np.array([1.0, 0, 0]) - n[0] * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)

    def heights(k):
        Pk, Mk, _ = moge(k)
        mk = np.load(CACHE / f"masks_{k}.npz")
        mov = cv2.dilate((mk["car"] | mk["cart"] | honeycomb(U[k], mk["cart"])).astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
        Qk = Pk[Mk & ~mov]
        nk, ck, _ = _plane_ransac(Qk[::7], 0.01 * float(np.median(Qk[:, 2])))
        if nk[2] > 0:
            nk = -nk
        hk = float(-(ck @ nk))
        return np.where(Mk, (hk + Pk @ nk) * h0 / hk, np.nan).astype(np.float32)

    H0 = heights(0)

    def to2d(px, hm):
        hh = np.array([hm[int(round(p[1])), int(round(p[0]))] for p in np.atleast_2d(px)])
        X = plane_point(np.atleast_2d(px).astype(float), K, n, c, hh)
        return np.c_[X @ e1, X @ e2]

    # The honeycomb's front column of photo targets at −50 ms (nothing has touched it): evenly spaced, 450 mm apart
    # (scale); its middle target is the face's centre (CG, below).
    m_cart0 = m0["cart"]
    cart_x = np.argwhere(m_cart0)[:, 1].min()
    discs = quartered_discs(U[0])
    cand = discs[(discs[:, 0] < cart_x) & (discs[:, 0] > cart_x - 300)]
    best = None
    for x in cand[:, 0]:
        col = cand[np.abs(cand[:, 0] - x) < 15]
        col = col[np.argsort(col[:, 1])]
        g = np.diff(col[:, 1])
        if len(col) >= 3 and np.ptp(g) < 0.12 * g.mean() and (best is None or col[:, 0].mean() < best[:, 0].mean()):
            best = col
    front_col = best
    q = to2d(front_col, H0)
    gaps = np.linalg.norm(np.diff(q, axis=0), axis=1)
    k = TRUTH["target_spacing_m"] / float(np.mean(gaps))
    mk = np.load(CACHE / f"masks_{k0}.npz")
    Hs = heights(k0)

    lines = lens_from_lines(gray[0], K, ~moving0)
    pl = np.array(lines["per_line"]) if lines["per_line"] else np.array([0.0])
    spread = float(np.subtract(*np.percentile(pl, [75, 25])))
    k1 = lines["k1"] if spread < 0.1 else 0.0
    und = radial(K, k1)

    # The barrier's front truss, seeded at impact: the orange steel itself (not the floor seen through it).
    hsv_k = cv2.cvtColor(U[k0], cv2.COLOR_BGR2HSV)
    orange = (hsv_k[..., 0] >= 3) & (hsv_k[..., 0] <= 18) & (hsv_k[..., 1] > 150) & (hsv_k[..., 2] > 90)
    truss = (cv2.dilate(orange.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & mk["cart"] & np.isfinite(Hs)
    frames_f, frames_b = list(range(k0, len(U))), list(range(k0, -1, -1))
    # Heights: each feature's own, or one plane at the truss's median height. The barrier runs straight on its
    # guide until impact, so whichever turns it less over the approach is the better geometry (no answers used).
    flat = np.full_like(Hs, float(np.nanmedian(Hs[truss])))
    trial = {}
    for name, hm in (("per-feature heights", Hs), ("one plane", flat)):
        r = track_rigid(gray, truss, frames_b, K, n, c, e1, e2, hm, und)
        ok = np.isfinite(r["theta"])
        trial[name] = (abs(float(np.degrees(r["theta"][ok][-1]))), hm, r)
    height_mode = min(trial, key=lambda kk: trial[kk][0])
    Hs = trial[height_mode][1]
    rb = trial[height_mode][2]
    rf = track_rigid(gray, truss, frames_f, K, n, c, e1, e2, Hs, und)
    th = np.full(len(U), np.nan)
    tr = np.full((len(U), 2), np.nan)
    for r, fr in ((rb, frames_b), (rf, frames_f)):
        for j, i in enumerate(fr):
            th[i], tr[i] = r["theta"][j], r["t"][j]

    # Direction of travel before impact (the barrier's forward axis): the truss's move over the approach.
    ok = np.flatnonzero(np.isfinite(th[:k0 - 2]))
    a, b = ok[0], k0 - 3
    fwd = (tr[b] - tr[a]) / np.linalg.norm(tr[b] - tr[a])
    # CG at −50 ms: 2.185 m behind the honeycomb's front face (its middle target) along the barrier's axis, on its
    # centre line; carried into the seed frame's coordinates through the pose at −50 ms (x_seed -> R x + t).
    face0 = q[len(q) // 2]
    cg0 = face0 - TRUTH["cg_behind_face_m"] / k * fwd
    Ra = np.array([[np.cos(th[0]), -np.sin(th[0])], [np.sin(th[0]), np.cos(th[0])]])
    cg_seed = Ra.T @ (cg0 - tr[0])
    cg = np.full((len(U), 2), np.nan)
    for i in range(len(U)):
        if np.isfinite(th[i]):
            R = np.array([[np.cos(th[i]), -np.sin(th[i])], [np.sin(th[i]), np.cos(th[i])]])
            cg[i] = (R @ cg_seed + tr[i]) * k

    def vel(p, half=4):
        v = np.full_like(p, np.nan)
        for i in range(half, len(p) - half):
            seg = p[i - half:i + half + 1]
            if np.isfinite(seg).all():
                v[i] = [np.polyfit(t[i - half:i + half + 1], seg[:, j], 1)[0] for j in (0, 1)]
        return v

    v = vel(cg)
    app = np.isfinite(cg).all(1) & (t >= -0.046) & (t <= -0.003)
    v_app = np.array([np.polyfit(t[app], cg[app, j], 1)[0] for j in (0, 1)])
    # In the barrier's own axes (forward, right), integrated from impact: what its accelerometers integrate to.
    dvX, dvY = np.full(len(U), np.nan), np.full(len(U), np.nan)
    sX = sY = 0.0
    for i in range(k0 + 1, len(U)):
        if not (np.isfinite(v[i]).all() and np.isfinite(v[i - 1]).all()):
            break
        R = np.array([[np.cos(th[i] - th[k0]), -np.sin(th[i] - th[k0])], [np.sin(th[i] - th[k0]), np.cos(th[i] - th[k0])]])
        f = R @ fwd
        right = np.array([f[1], -f[0]])                                  # seen from above, right of forward
        dv = v[i] - v[i - 1]
        sX += dv @ f
        sY += dv @ right
        dvX[i], dvY[i] = sX, sY
    ts, axs = nhtsa_sensors.speed_change(*nhtsa_sensors.curve(10072, 267), t_end=0.19)
    _, ays = nhtsa_sensors.speed_change(*nhtsa_sensors.curve(10072, 268), t_end=0.19)
    rows = []
    for ms in (10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 190):
        i = k0 + ms
        rows.append({"t_ms": ms, "dvX_kmh": None if np.isnan(dvX[i]) else round(dvX[i] * 3.6, 1),
                     "dvX_true": round(float(np.interp(ms / 1000, ts, axs)) * 3.6, 1),
                     "dvY_kmh": None if np.isnan(dvY[i]) else round(dvY[i] * 3.6, 1),
                     "dvY_true": round(float(np.interp(ms / 1000, ts, ays)) * 3.6, 1),
                     "turn_deg": None if np.isnan(th[i]) else round(float(np.degrees(th[i] - th[k0])), 1)})
    res = {"camera": "No. 12 Overhead of OMDB View, 28 mm, 1,000 fps", "frames": len(U),
           "scale_targets_px": front_col.round(1).tolist(), "target_gaps_ai": gaps.round(4).tolist(), "scale": round(k, 4),
           "floor_tilt_deg": round(float(np.degrees(np.arccos(abs(n[2])))), 1),
           "lens": {"k1": k1, "lines_per_line": lines["per_line"], "lines_spread": round(spread, 3),
                    "rule": "lines only if the middle half of per-line values spans < 0.1; else none"},
           "heights": {"used": height_mode, "turn_over_approach_deg": {kk: round(v[0], 2) for kk, v in trial.items()}},
           "tracked_ms": [round(float(t[np.isfinite(th)][0] * 1000)), round(float(t[np.isfinite(th)][-1] * 1000))],
           "approach_kmh": round(float(np.linalg.norm(v_app)) * 3.6, 2), "approach_frames": int(app.sum()),
           "truth_kmh": TRUTH["speed_kmh"], "barrier_axes": rows,
           "curves": {"t": t.tolist(), "cg": np.round(cg, 4).tolist(), "turn": np.round(th, 5).tolist(),
                      "dvX": np.round(dvX, 4).tolist(), "dvY": np.round(dvY, 4).tolist()}}
    print(json.dumps({kk: vv for kk, vv in res.items() if kk != "curves"}, indent=1))
    (ROOT / "benchmark" / "omdb10072.json").write_text(json.dumps(res, default=float))
    return res


if __name__ == "__main__":
    main()
