"""A two-body event: NHTSA test 8555, a 2008 Ford F-150 (2,673.8 kg) into the rear of a standing 2004 Jeep Liberty
(1,926.6 kg) at 69.1 km/h, measured the hybrid way (physlens/hybrid.py) from the overhead high-speed camera
(No. 8, 1,000 fps; NHTSA's transcode shows every frame twice, so frames are taken once per burned-in clock reading).

  floor   a plane fitted to MoGe-3's points of the floor (frame at −200 ms, the Jeep alone)
  scale   the Jeep's published width (1.815 m) against its width in MoGe-3's points at −200 ms
  planes  each tracked part moves in a plane parallel to the floor at its own height (MoGe-3's points of the part)
  track   F-150: its hood before impact (one plane); after impact its cab (roof and the hood's rear, not the
          glass), seeded at +60 ms and followed both ways, every feature at its own MoGe-3 height;
          Jeep: the front 60% of its roof (away from the struck rear); every frame
  lens    radial term from straight floor lines (the physics fit, constant approach speed, reported beside it)
  check   both speed curves against the cars' accelerometers, and the momentum each car gains and loses
Clock 0 ms is impact. Usage: PYTHONPATH=backend backend/.venv/bin/python benchmark/rear8555.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "benchmark"))
from dotenv import load_dotenv   # noqa: E402
load_dotenv(ROOT / "backend" / ".env")
from physlens.hybrid import along_from_pairs, fit_distortion, plane_point, radial, track_part   # noqa: E402
from physlens.moving_camera import _plane_ransac   # noqa: E402
import nhtsa_sensors   # noqa: E402

VIDEO = ROOT / "datasets" / "nhtsa" / "8555" / "cam08_overhead.mp4"
CACHE = ROOT / "records" / "real_cache" / "rear8555"
T0 = -200                                    # ms at the first frame
TRUTH = {"speed_kmh": 69.1, "m_truck": 2673.8, "m_jeep": 1926.6, "jeep_width_m": 1.815, "source": "NHTSA test 8555"}


def unique_frames():
    """One frame per clock reading (the burned-in clock box, binarised, changes once per millisecond)."""
    f = CACHE / "frames_u8.npy"
    if f.exists():
        return np.load(f)
    cap = cv2.VideoCapture(str(VIDEO))
    F = []
    while True:
        ok, x = cap.read()
        if not ok:
            break
        F.append(x)
    clk = np.array([cv2.cvtColor(x[2:38, 2:120], cv2.COLOR_BGR2GRAY) > 128 for x in F])
    new = np.r_[True, [(clk[i] != clk[i - 1]).sum() > 30 for i in range(1, len(F))]]
    U = np.array([F[i] for i in np.flatnonzero(new)])
    np.save(f, U)
    return U


def moge(k):
    z = np.load(CACHE / f"moge_{k}.npz")
    D, M, K = z["d"], z["m"] & np.isfinite(z["d"]), z["K"]
    h, w = D.shape
    ys, xs = np.mgrid[0:h, 0:w]
    return np.dstack([(xs - K[0, 2]) * D / K[0, 0], (ys - K[1, 2]) * D / K[1, 1], D]), M, K


def red_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] < 10) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 90) & (hsv[..., 2] > 60)
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0


# Straight floor lines in the frame at −200 ms (slab seams and edges; rough endpoints, then refined per column).
LINES = {"seam_465": ("h", 462, 468, 70, 735, True), "edge_650": ("h", 652, 646, 15, 515, False),
         "edge_145": ("h", 138, 150, 95, 735, False), "v_65": ("v", 68, 64, 40, 215, False),
         "v_55": ("v", 57, 53, 315, 465, False)}


def lens_from_lines(gray0, K, k1s=np.round(np.arange(-0.4, 0.41, 0.02), 2)):
    """The radial term that makes straight floor lines straight: each line's own best term, then the median over
    lines (one bent or mis-picked line can't drag it). Returns the term and each line's."""
    g = cv2.GaussianBlur(gray0.astype(float), (5, 5), 0)
    gy = np.abs(cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=5))
    gx = np.abs(cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=5))

    def peak(prof):
        j = int(np.argmax(prof))
        if not 0 < j < len(prof) - 1:
            return None
        a, b, c = prof[j - 1:j + 2]
        return j + (0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c else 0.0)

    def sample(kind, p0, p1, a, b, dark):
        pts, band = [], 10
        for s in range(a, b, 3):
            q = int(round(p0 + (p1 - p0) * (s - a) / (b - a)))
            prof = (-g[q - band:q + band, s] if dark else gy[q - band:q + band, s]) if kind == "h" else gx[s, q - band:q + band]
            j = peak(prof)
            if j is not None:
                pts.append((s, q - band + j) if kind == "h" else (q - band + j, s))
        return np.array(pts, float)

    def rms(p):
        c = p.mean(0)
        r = (p - c) @ np.linalg.svd(p - c, full_matrices=False)[2][-1]
        p = p[np.abs(r) < 3 * np.median(np.abs(r)) + 0.5]
        c = p.mean(0)
        return float(np.sqrt(np.mean(((p - c) @ np.linalg.svd(p - c, full_matrices=False)[2][-1]) ** 2)))

    per = {}
    for name, spec in LINES.items():
        p = sample(*spec)
        per[name] = float(k1s[int(np.argmin([rms(radial(K, k1)(p)) for k1 in k1s]))])
    return float(np.median(list(per.values()))), per


def main():
    U = unique_frames()
    gray = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in U]
    h, w = gray[0].shape
    ys, xs = np.mgrid[0:h, 0:w]
    inner = (xs - w / 2) ** 2 + (ys - h / 2) ** 2 < (0.44 * w) ** 2       # inside the lens's vignette
    t = (np.arange(len(U)) + T0) / 1000.0
    k0 = -T0                                                             # frame index of impact

    # Floor and scale from the frame with the Jeep alone.
    P, M, K = moge(0)
    masks = np.load(CACHE / "masks_0.npz")
    jeep0 = masks["jeep"] | masks["truck"] if masks["jeep"].sum() == 0 else masks["jeep"]
    cars = cv2.dilate(jeep0.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    Q = P[M & inner & ~cars]
    n, c, _ = _plane_ransac(Q[::5], 0.01 * float(np.median(Q[:, 2])))
    if n[2] > 0:
        n = -n                                                           # facing the camera
    h0 = float(-(c @ n))
    ax = np.array([1.0, 0, 0]) - n[0] * n
    ax /= np.linalg.norm(ax)
    side = np.cross(n, ax)
    J = P[(cv2.erode(jeep0.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & M]
    W_ai = float(np.subtract(*np.percentile(J @ side, [99.5, 0.5])))
    k = TRUTH["jeep_width_m"] / W_ai

    def plane_at(height_ai):
        """The plane parallel to the floor, height_ai above it (frame-0 MoGe units)."""
        return n, c + height_ai * n                                       # n faces the camera

    def part_height(kf, region):
        """A part's height above the floor in frame kf's MoGe points, carried to frame 0's units by the camera's
        height above the floor (the same camera over the same floor in every frame)."""
        Pk, Mk, _ = moge(kf)
        mz = np.load(CACHE / f"masks_{kf}.npz")
        cars_k = cv2.dilate((mz["jeep"] | mz["truck"]).astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
        Qk = Pk[Mk & inner & ~cars_k]
        nk, ck, _ = _plane_ransac(Qk[::5], 0.01 * float(np.median(Qk[:, 2])))
        if nk[2] > 0:
            nk = -nk
        hk = float(-(ck @ nk))
        hts = hk + Pk[region & Mk] @ nk
        return float(np.median(hts)) * h0 / hk

    # The Jeep: the front 60% of its roof (right of its rear 40%), seeded at −200 ms.
    jx = np.argwhere(jeep0)[:, 1]
    jeep_roof = (cv2.erode(jeep0.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & (xs > np.percentile(jx, 40))
    hj = float(np.median(h0 + P[jeep_roof & M] @ n))
    nj, cj = plane_at(hj)

    def carried(m0, inside=inner, keep_share=0.4):
        """m0 carried by the part's image shift, within the lens's clear circle; empty once less than keep_share
        of it is left in the picture (no features on whatever is left behind)."""
        def at(i, sh):
            m = (cv2.warpAffine(m0.astype(np.uint8), np.float32([[1, 0, sh[0]], [0, 1, sh[1]]]), (w, h),
                                flags=cv2.INTER_NEAREST) > 0) & inside
            return m if m.sum() >= keep_share * m0.sum() else np.zeros_like(m)
        return at

    jeep_region = carried(jeep_roof, keep_share=0.2)

    def heights(kf):
        """Each pixel's height above the floor in frame kf's MoGe points, in frame 0's units (scaled by the
        camera's height above the floor)."""
        Pk, Mk, _ = moge(kf)
        mz = np.load(CACHE / f"masks_{kf}.npz")
        cars_k = cv2.dilate((mz["jeep"] | mz["truck"]).astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
        Qk = Pk[Mk & inner & ~cars_k]
        nk, ck, _ = _plane_ransac(Qk[::5], 0.01 * float(np.median(Qk[:, 2])))
        if nk[2] > 0:
            nk = -nk
        hk = float(-(ck @ nk))
        return np.where(Mk, (hk + Pk @ nk) * h0 / hk, np.nan).astype(np.float32)

    def carried_map(hm):
        return lambda i, sh: cv2.warpAffine(hm, np.float32([[1, 0, sh[0]], [0, 1, sh[1]]]), (w, h),
                                            flags=cv2.INTER_NEAREST, borderValue=np.nan)

    def track_both(seed, lo, hi, region, hm, und_):
        """Seeded at frame `seed`, followed back to `lo` and on to `hi`, each feature at its own height:
        positions (m, relative to the seed) and features, frames lo..hi."""
        rf = track_part(gray, region, seed, hi, K, n, c, d, undistort=und_, height_at=carried_map(hm))
        rb = track_part(gray[seed:lo - 1 if lo > 0 else None:-1], region, 0, seed - lo, K, n, c, d, undistort=und_,
                        height_at=carried_map(hm))
        return np.r_[rb["along_m"][:0:-1], rf["along_m"]] * k, np.r_[rb["features"][:0:-1], rf["features"]], rf

    # The F-150: red paint on its rows that enters from the left (not the Jeep's red logo or lamps). Each part is
    # seeded once and carried by its own image motion, as the Jeep's roof is: before impact its hood (the front is
    # painted white), after impact red paint more than 1.7 m behind its front (the cab and bed; the hood crushes).
    def truck_red(i):
        m = red_mask(U[i]) & inner
        m[ys > 420] = False                                              # its rows in the picture
        nl, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        keep = [b for b in range(1, nl) if st[b, cv2.CC_STAT_AREA] > 400 and st[b, cv2.CC_STAT_LEFT] < 300]
        return np.isin(lab, keep)

    px_per_m = float(np.percentile(np.argwhere(jeep0)[:, 0], 99) - np.percentile(np.argwhere(jeep0)[:, 0], 1)) / TRUTH["jeep_width_m"]

    def cab_at(i):
        m = cv2.erode(truck_red(i).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        return m & (xs < np.percentile(np.argwhere(m)[:, 1], 99.5) - 1.7 * px_per_m) if m.any() else m

    first = next(i for i in range(len(U)) if truck_red(i).sum() > 3000)
    hood0 = cv2.erode(truck_red(first).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    hood_region = carried(hood0)
    hood_h = part_height(185, cv2.erode(truck_red(185).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    nh, ch_ = plane_at(hood_h)

    # Direction of travel: the hood's median plane displacement over the approach.
    rh = track_part(gray, hood_region, first, k0 - 1, K, nh, ch_, ax)
    steps = []
    for pr in rh["pairs"]:
        if pr is not None:
            steps.append(np.median(plane_point(pr[1], K, nh, ch_) - plane_point(pr[0], K, nh, ch_), axis=0))
    d = np.sum(steps, axis=0)
    d -= (d @ n) * n
    d /= np.linalg.norm(d)
    tt = t[first:k0]
    steady = (tt >= -0.1) & (tt <= -0.003)
    # Lens: from the straight floor lines (a calibration of the camera, independent of the crash). The physics fit
    # (constant approach speed) is reported beside it; the hood's approach covers only the picture's left edge.
    k1, k1_lines = lens_from_lines(gray[0], K)
    phys = fit_distortion(rh["pairs"], K, nh, ch_, d, tt, steady)
    und = radial(K, k1)
    lens = {"k1": k1, "from": "straight floor lines", "per_line": k1_lines,
            "physics_fit_k1": phys["k1"], "physics_fit_relative_residual": phys["relative_residual"]}

    def speed(x, tt, feats, half=4):
        """Speed (m/s) at each frame: a straight line through ±half frames of position; none where the part
        wasn't tracked (no features) in that window."""
        v = np.full(len(x), np.nan)
        for i in range(half, len(x) - half):
            if (feats[i - half:i + half + 1] > 0).all():
                v[i] = np.polyfit(tt[i - half:i + half + 1], x[i - half:i + half + 1], 1)[0]
        return v

    # The approach: the hood, one nearly flat surface, on its plane.
    xh = along_from_pairs(rh["pairs"], K, nh, ch_, d, und) * k
    v_app = float(np.polyfit(tt[steady], xh[steady], 1)[0])
    v_app_raw = float(np.polyfit(tt[steady], (along_from_pairs(rh["pairs"], K, nh, ch_, d) * k)[steady], 1)[0])

    # After impact. The Jeep's roof, from −200 ms on.
    xj, fj, rj = track_both(0, 0, len(U) - 1, jeep_region, heights(0), und)
    # The F-150's cab (roof, windscreen, the hood's rear: rigid in this crash), seeded at +60 ms where MoGe-3 gives
    # each pixel its height, more than 1.5 m behind the Jeep's rear (the F-150's front crushes into it), then
    # followed back to impact and on; every feature moves in the plane at its own height.
    cab_seed = 260
    Pk, Mk, _ = moge(cab_seed)
    vehicles = np.load(CACHE / f"masks_{cab_seed}.npz")["truck"]             # SAM: both vehicles, one mask
    cars_k = cv2.dilate(vehicles.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    Qk = Pk[Mk & inner & ~cars_k]
    nk, ck, _ = _plane_ransac(Qk[::5], 0.01 * float(np.median(Qk[:, 2])))
    if nk[2] > 0:
        nk = -nk
    hk = float(-(ck @ nk))
    hmap = heights(cab_seed)
    jeep_rear = np.percentile(np.argwhere(jeep0)[:, 1], 1) + rj["shift_px"][cab_seed][0]
    # Not the windscreen or windows: glass shows the room's reflection (it doesn't move with the truck) and its
    # depth is unreliable. Dark, unsaturated pixels, grown a little to take in the glass's edges.
    hsv = cv2.cvtColor(U[cab_seed], cv2.COLOR_BGR2HSV)
    glass = cv2.dilate(((hsv[..., 2] < 90) & (hsv[..., 1] < 120)).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    wide = (xs - w / 2) ** 2 + (ys - h / 2) ** 2 < (0.47 * w) ** 2
    cab0 = (cv2.erode(vehicles.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & (xs < jeep_rear - 1.5 * px_per_m) \
        & (ys < 420) & (hmap * k > 0.9) & wide & ~glass
    cab_region = carried(cab0, wide)

    xc, fc, _ = track_both(cab_seed, k0, len(U) - 1, cab_region, hmap, und)
    cab_h = float(np.nanmedian(hmap[cab0]))
    tc, tj = t[k0:], t[0:]
    vc, vj = speed(xc, tc, fc), speed(xj, tj, fj)

    # Truth: speed curves from the accelerometers (F-150 curve 4, CG X redundant; Jeep curves 7 and 10 averaged).
    st, sv1 = nhtsa_sensors.speed_change(*nhtsa_sensors.curve(8555, 4), t_end=0.3)
    _, sv2a = nhtsa_sensors.speed_change(*nhtsa_sensors.curve(8555, 7), t_end=0.3)
    _, sv2b = nhtsa_sensors.speed_change(*nhtsa_sensors.curve(8555, 10), t_end=0.3)
    v_truck_true = TRUTH["speed_kmh"] / 3.6 + sv1
    v_jeep_true = (sv2a + sv2b) / 2

    def at(tq, tt, v):
        return float(np.interp(tq, tt, v))

    rows = []
    for ms in (0, 20, 40, 60, 80, 100, 120, 150, 200, 250, 300):
        tq = ms / 1000
        rows.append({"t_ms": ms, "truck_kmh": round(at(tq, tc, vc) * 3.6, 1) if np.isfinite(at(tq, tc, vc)) else None,
                     "truck_true_kmh": round(at(tq, st, v_truck_true) * 3.6, 1),
                     "jeep_kmh": round(at(tq, tj, vj) * 3.6, 1) if np.isfinite(at(tq, tj, vj)) else None, "jeep_true_kmh": round(at(tq, st, v_jeep_true) * 3.6, 1)})
    # Momentum: the speed each car loses / gains by 100–200 ms (the mean of that window; the Jeep leaves the
    # picture at about 230 ms), from the video alone.
    win = lambda tt: (tt >= 0.10) & (tt <= 0.20)
    dv_truck = float(np.nanmean(vc[win(tc)])) - v_app
    dv_jeep = float(np.nanmean(vj[win(tj)]))
    res_cov = {"truck_tracked_ms": [round(float(tc[np.isfinite(vc)][0] * 1000)), round(float(tc[np.isfinite(vc)][-1] * 1000))],
               "jeep_tracked_ms": [round(float(tj[np.isfinite(vj)][0] * 1000)), round(float(tj[np.isfinite(vj)][-1] * 1000))]}
    res = {**res_cov, "camera": "No. 8 overhead, 1,000 fps", "frames": len(U), "scale_from_jeep_width": round(k, 4),
           "heights_m": {"jeep_roof": round(hj * k, 3), "truck_hood": round(hood_h * k, 3), "truck_cab": round(cab_h * k, 3)},
           "floor_tilt_deg": round(float(np.degrees(np.arccos(abs(n[2])))), 1), "lens": lens,
           "features_median": {"hood": int(np.median(rh["features"])), "cab": int(np.median(fc[fc > 0])),
                               "jeep": int(np.median(fj[fj > 0]))},
           "approach_kmh": round(v_app * 3.6, 2), "approach_kmh_without_lens_fix": round(v_app_raw * 3.6, 2),
           "approach_frames": [round(float(tt[steady][0] * 1000)), round(float(tt[steady][-1] * 1000))],
           "truth_kmh": TRUTH["speed_kmh"], "speeds": rows,
           "dv_truck_kmh": round(dv_truck * 3.6, 1), "dv_jeep_kmh": round(dv_jeep * 3.6, 1),
           "dv_truth_kmh": {"truck": round(float(np.mean(sv1[win(st)])) * 3.6, 1),
                            "jeep": round(float(np.mean(v_jeep_true[win(st)])) * 3.6, 1)},
           "momentum_lost_by_truck": round(-dv_truck * TRUTH["m_truck"]), "momentum_gained_by_jeep": round(dv_jeep * TRUTH["m_jeep"]),
           "curves": {"t_hood": tt.tolist(), "x_hood": xh.round(4).tolist(), "t_cab": tc.tolist(), "v_cab": np.round(vc, 3).tolist(),
                      "t_jeep": tj.tolist(), "v_jeep": np.round(vj, 3).tolist(), "t_true": st.tolist(),
                      "v_truck_true": v_truck_true.round(3).tolist(), "v_jeep_true": v_jeep_true.round(3).tolist()}}
    res["momentum_balance"] = round(res["momentum_gained_by_jeep"] / res["momentum_lost_by_truck"], 3)
    res["momentum_balance_from_accelerometers"] = round(
        res["dv_truth_kmh"]["jeep"] * TRUTH["m_jeep"] / (-res["dv_truth_kmh"]["truck"] * TRUTH["m_truck"]), 3)
    print(json.dumps({kk: vv for kk, vv in res.items() if kk != "curves"}, indent=1))
    (ROOT / "benchmark" / "rear8555.json").write_text(json.dumps(res, default=float))
    return res


if __name__ == "__main__":
    main()
