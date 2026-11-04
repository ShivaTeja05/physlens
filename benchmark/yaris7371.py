"""Two bodies that turn: NHTSA test 7371, a 2010 Toyota Yaris (1,345 kg) at 113.0 km/h into a standing 2010 Yaris
(1,345 kg), 15°, 50% overlap, measured from the overhead high-speed camera ("02 Overhead Wide", 1,000 fps).

  floor   a plane fitted to MoGe-3's points of the floor (frame at −18.6 ms)
  scale   the struck Yaris's published width (1.695 m) against its width in MoGe-3's points
  lens    radial term from straight floor lines (hand-picked seams, each refined across its length; median)
  bodies  each car is a rigid body sliding and turning on the floor (hybrid.track_rigid): its features meet the
          plane at their own MoGe-3 heights, the move per millisecond is the rigid move that best fits them.
          Struck car: its rear 60% (away from the impact), seeded at −18.6 ms. Bullet car: before impact all its
          visible body; after impact more than 1.5 m behind its front, seeded at +38.4 ms and followed both ways.
  CG      each car's centre-of-gravity target (the quartered disc on its roof), carried by the body's pose
  check   the bullet's approach against the speed trap; the struck car's acceleration in its own (turning) axes,
          integrated, against its CG accelerometers; momentum (equal masses) from the video alone
The clock's first three frames repeat −18.62 ms; from there one frame per millisecond. Usage:
PYTHONPATH=backend backend/.venv/bin/python benchmark/yaris7371.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "benchmark"))
from physlens.hybrid import lens_from_lines, plane_point, radial, track_rigid   # noqa: E402
from physlens.moving_camera import _plane_ransac   # noqa: E402
import nhtsa_sensors   # noqa: E402

CACHE = ROOT / "records" / "real_cache" / "yaris7371"
T0 = -18.62                                   # ms at frames_u8[0] (the video's 4th frame)
TRUTH = {"speed_kmh": 113.0, "mass_kg": 1345, "width_m": 1.695, "length_m": 4.28, "source": "NHTSA test 7371"}
# Straight floor seams in the first frame: rough ends (pixels), refined across their length.
SEAMS = [(393, 20, 390, 300), (395, 806, 1450, 796), (400, 303, 690, 305), (1463, 583, 1468, 800),
         (922, 35, 920, 285), (1126, 83, 1407, 99), (115, 211, 344, 220)]


def moge(k):
    z = np.load(CACHE / f"moge_{k}.npz")
    D, M, K = z["d"], z["m"] & np.isfinite(z["d"]), z["K"]
    h, w = D.shape
    ys, xs = np.mgrid[0:h, 0:w]
    return np.dstack([(xs - K[0, 2]) * D / K[0, 0], (ys - K[1, 2]) * D / K[1, 1], D]), M, K


def cars(k):
    """SAM 3.1's two car masks in frame k: (struck, bullet) — the struck car is the one further right."""
    z = np.load(CACHE / f"masks_{k}.npz")
    m = sorted((z[f] for f in z.files), key=lambda a: -np.argwhere(a)[:, 1].mean())
    return m[0], m[1]


def floor(P, M, clear, vehicles):
    away = cv2.dilate(vehicles.astype(np.uint8), np.ones((41, 41), np.uint8)) == 0
    Q = P[M & clear & away]
    n, c, _ = _plane_ransac(Q[::7], 0.01 * float(np.median(Q[:, 2])))
    return (-n, c) if n[2] > 0 else (n, c)                               # n faces the camera


def cg_marker(img, mask):
    """The CG target, a disc in four quadrants (two dark, two light): the circle in the car whose ring alternates
    most strongly twice around (the second harmonic). Returns its centre (pixels) and radius."""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    cs = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=40, param1=120, param2=25, minRadius=25, maxRadius=60)
    best = None
    ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    for x, y, r in ([] if cs is None else cs[0]):
        if not mask[int(min(y, mask.shape[0] - 1)), int(min(x, mask.shape[1] - 1))]:
            continue
        ring = np.mean([g[(y + f * r * np.sin(ang)).astype(int).clip(0, g.shape[0] - 1),
                          (x + f * r * np.cos(ang)).astype(int).clip(0, g.shape[1] - 1)].astype(float)
                        for f in (0.4, 0.6, 0.8)], axis=0)
        sp = np.abs(np.fft.rfft(ring - ring.mean()))
        score = sp[2] / (sp[1:].sum() + 1e-9) * np.ptp(ring)
        if best is None or score > best[0]:
            best = (score, np.array([x, y], float), float(r))
    return best[1], best[2]


def main(k1_override=None):
    U = np.load(CACHE / "frames_u8.npy")
    gray = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in U]
    h, w = gray[0].shape
    ys, xs = np.mgrid[0:h, 0:w]
    t = (np.arange(len(U)) + T0) / 1000.0
    k_imp = int(round(-T0))                                              # first frame at or after 0 ms (+0.38)
    clear = cv2.erode((gray[0] > 25).astype(np.uint8), np.ones((61, 61), np.uint8)) > 0   # inside the vignette

    P, M, K = moge(0)
    struck0, bullet0 = cars(0)
    n, c = floor(P, M, clear, struck0 | bullet0)
    h0 = float(-(c @ n))
    e1 = np.array([1.0, 0, 0]) - n[0] * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)                                                 # (e1, e2) seen from above: anticlockwise

    def heights(k):
        """Each pixel's height above the floor in frame k's MoGe points, in frame 0's units."""
        Pk, Mk, _ = moge(k)
        nk, ck = floor(Pk, Mk, clear, np.logical_or(*cars(k)))
        hk = float(-(ck @ nk))
        return np.where(Mk, (hk + Pk @ nk) * h0 / hk, np.nan).astype(np.float32)

    H0 = heights(0)

    def to2d(px, hm):
        off = hm[int(px[1]), int(px[0])]
        X = plane_point(np.asarray(px, float)[None], K, n, c, off)[0]
        return np.array([X @ e1, X @ e2])

    # Scale: the struck car's width across its long axis, all its points, on the floor plane.
    Sp = P[(cv2.erode(struck0.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & M]
    S2 = np.c_[Sp @ e1, Sp @ e2]
    axis = np.linalg.svd(S2 - S2.mean(0), full_matrices=False)[2][0]
    across = np.array([-axis[1], axis[0]])
    W_ai = float(np.subtract(*np.percentile(S2 @ across, [99.5, 0.5])))
    L_ai = float(np.subtract(*np.percentile(S2 @ axis, [99.5, 0.5])))
    k = TRUTH["width_m"] / W_ai

    lines = lens_from_lines(gray[0], K, np.ones_like(gray[0], bool), segments=SEAMS)
    k1 = lines["k1"] if k1_override is None else k1_override
    und = radial(K, k1)

    hsv = cv2.cvtColor(U[0], cv2.COLOR_BGR2HSV)

    def no_glass(img):
        s = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return ~(cv2.dilate(((s[..., 2] < 90) & (s[..., 1] < 120)).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)

    def front_is(px_mask, toward_x):
        """Along-axis coordinate (pixels) of the car's points, the front being the end nearest image x toward_x."""
        pts = np.argwhere(px_mask)[:, ::-1].astype(float)
        u = np.linalg.svd(pts - pts.mean(0), full_matrices=False)[2][0]
        a = (pts - pts.mean(0)) @ u
        if (pts[np.argmax(a)][0] - toward_x) ** 2 > (pts[np.argmin(a)][0] - toward_x) ** 2:
            u = -u                                                       # +u points to the front
        return u, pts.mean(0)

    # The struck car: its rear 60%, seeded at −18.6 ms. Its front faces the bullet (left).
    u_s, m_s = front_is(struck0, 0)
    a_s = ((np.c_[xs.ravel(), ys.ravel()] - m_s) @ u_s).reshape(h, w)
    span = np.percentile(a_s[struck0], [1, 99])
    struck_rear = (cv2.erode(struck0.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & no_glass(U[0]) \
        & (a_s < span[1] - 0.4 * (span[1] - span[0])) & clear
    rs = track_rigid(gray, struck_rear, list(range(0, len(U))), K, n, c, e1, e2, H0, und)

    # The bullet before impact: all its visible body (nothing crushed yet).
    bullet_app = (cv2.erode(bullet0.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & no_glass(U[0]) & clear
    rb0 = track_rigid(gray, bullet_app, list(range(0, k_imp)), K, n, c, e1, e2, H0, und)
    # The bullet after impact: seeded at +38.4 ms, more than 1.5 m behind its front, followed both ways.
    ks = 57
    _, bullet57 = cars(ks)
    H57 = heights(ks)
    u_b, m_b = front_is(bullet57, w)
    a_b = ((np.c_[xs.ravel(), ys.ravel()] - m_b) @ u_b).reshape(h, w)
    px_per_m = float(np.subtract(*np.percentile(np.argwhere(struck0)[:, 1], [99, 1]))) / (L_ai * k) \
        if False else None
    front_b = np.percentile(a_b[bullet57], 99)
    m_per_px = TRUTH["width_m"] / float(np.subtract(*np.percentile(((np.c_[xs.ravel(), ys.ravel()] - m_s) @ np.array([-u_s[1], u_s[0]])).reshape(h, w)[struck0], [99.5, 0.5])))
    bullet_rear = (cv2.erode(bullet57.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & no_glass(U[ks]) \
        & (a_b < front_b - 1.5 / m_per_px) & clear
    rb_f = track_rigid(gray, bullet_rear, list(range(ks, len(U))), K, n, c, e1, e2, H57, und)
    rb_b = track_rigid(gray, bullet_rear, list(range(ks, k_imp - 1, -1)), K, n, c, e1, e2, H57, und)

    def cg_path(r, frames, seed_img, seed_mask, hm):
        """The CG target's 2D position (metres) per frame, through the body's pose."""
        px, rad = cg_marker(seed_img, seed_mask)
        q0 = to2d(px, hm) if np.isfinite(hm[int(px[1]), int(px[0])]) else None
        if q0 is None:
            disc = (xs - px[0]) ** 2 + (ys - px[1]) ** 2 < (0.8 * rad) ** 2
            hm = np.where(np.isfinite(hm), hm, np.nanmedian(hm[disc]))
            q0 = to2d(px, hm)
        out = np.full((len(U), 2), np.nan)
        for j, i in enumerate(frames):
            th, tt = r["theta"][j], r["t"][j]
            if np.isfinite(th):
                R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                out[i] = (R @ q0 + tt) * k
        return out, px

    cg_s, px_s = cg_path(rs, list(range(0, len(U))), U[0], struck0, H0)
    cg_b0, px_b = cg_path(rb0, list(range(0, k_imp)), U[0], bullet0, H0)
    cgf, _ = cg_path(rb_f, list(range(ks, len(U))), U[ks], bullet57, H57)
    cgb, _ = cg_path(rb_b, list(range(ks, k_imp - 1, -1)), U[ks], bullet57, H57)
    cg_b1 = np.where(np.isfinite(cgf), cgf, cgb)

    def vel(p, half=4):
        v = np.full_like(p, np.nan)
        for i in range(half, len(p) - half):
            seg = p[i - half:i + half + 1]
            if np.isfinite(seg).all():
                v[i] = [np.polyfit(t[i - half:i + half + 1], seg[:, j], 1)[0] for j in (0, 1)]
        return v

    v_s, v_b1 = vel(cg_s), vel(cg_b1)
    app = np.isfinite(cg_b0).all(1) & (t <= -0.002)
    v_b0 = np.array([np.polyfit(t[app], cg_b0[app, j], 1)[0] for j in (0, 1)])

    # The struck car's heading (its front) and right side, turning with it; acceleration in those axes,
    # integrated from impact: what its CG accelerometers integrate to.
    f0 = np.array([u_s @ [1, 0], -(u_s @ [0, 1])])                      # image (x right, y down) -> (e1, e2)
    f0 /= np.linalg.norm(f0)
    th_s = rs["theta"]
    dvX, dvY = np.full(len(U), np.nan), np.full(len(U), np.nan)
    sX = sY = 0.0
    started = False
    for i in range(1, len(U)):
        if t[i] < 0:
            continue
        if not (np.isfinite(v_s[i]).all() and np.isfinite(v_s[i - 1]).all() and np.isfinite(th_s[i])):
            if started:
                break
            continue
        started = True
        R = np.array([[np.cos(th_s[i]), -np.sin(th_s[i])], [np.sin(th_s[i]), np.cos(th_s[i])]])
        f = R @ f0
        right = np.array([f[1], -f[0]])                                  # seen from above, right of forward
        dv = v_s[i] - v_s[i - 1]
        sX += dv @ f
        sY += dv @ right
        dvX[i], dvY[i] = sX, sY

    ts, ax_ = nhtsa_sensors.speed_change(*nhtsa_sensors.curve(7371, 275), t_end=0.3)
    _, ay_ = nhtsa_sensors.speed_change(*nhtsa_sensors.curve(7371, 276), t_end=0.3)
    rows = []
    for ms in (10, 20, 30, 40, 50, 60, 80, 100, 120, 150):
        i = int(np.argmin(np.abs(t - ms / 1000)))
        rows.append({"t_ms": ms, "dvX_kmh": None if np.isnan(dvX[i]) else round(dvX[i] * 3.6, 1),
                     "dvX_true": round(float(np.interp(ms / 1000, ts, ax_)) * 3.6, 1),
                     "dvY_kmh": None if np.isnan(dvY[i]) else round(dvY[i] * 3.6, 1),
                     "dvY_true": round(float(np.interp(ms / 1000, ts, ay_)) * 3.6, 1),
                     "struck_turn_deg": None if np.isnan(th_s[i]) else round(float(np.degrees(th_s[i])), 1)})
    # Momentum, equal masses: the bullet's velocity before = the two velocities after (a window both are seen in).
    both = np.isfinite(v_s).all(1) & np.isfinite(v_b1).all(1) & (t >= 0.09)
    win = np.flatnonzero(both)[:30]
    v_after = (v_s[win].mean(0) + v_b1[win].mean(0)) if len(win) else np.full(2, np.nan)
    res = {"camera": "02 Overhead Wide, 1,000 fps", "frames": len(U), "scale_from_width": round(k, 4),
           "length_from_width_scale_m": round(L_ai * k, 3), "true_length_m": TRUTH["length_m"],
           "floor_tilt_deg": round(float(np.degrees(np.arccos(abs(n[2])))), 1),
           "lens": {"k1": k1, "from": "straight floor lines" if k1_override is None else "given",
                    "per_line": lines["per_line"]},
           "cg_markers_px": {"struck": px_s.round(1).tolist(), "bullet": px_b.round(1).tolist()},
           "tracked_ms": {"struck": [round(float(t[np.isfinite(cg_s).all(1)][0] * 1000)), round(float(t[np.isfinite(cg_s).all(1)][-1] * 1000))],
                          "bullet_after": [round(float(t[np.isfinite(cg_b1).all(1)][0] * 1000)), round(float(t[np.isfinite(cg_b1).all(1)][-1] * 1000))]},
           "approach_kmh": round(float(np.linalg.norm(v_b0)) * 3.6, 2), "approach_frames": int(app.sum()),
           "truth_kmh": TRUTH["speed_kmh"], "struck_axes": rows,
           "momentum": {"window_ms": [round(float(t[win[0]] * 1000)), round(float(t[win[-1]] * 1000))] if len(win) else None,
                        "bullet_before_kmh": round(float(np.linalg.norm(v_b0)) * 3.6, 1),
                        "struck_after_kmh": round(float(np.linalg.norm(v_s[win].mean(0))) * 3.6, 1) if len(win) else None,
                        "bullet_after_kmh": round(float(np.linalg.norm(v_b1[win].mean(0))) * 3.6, 1) if len(win) else None,
                        "sum_after_kmh": round(float(np.linalg.norm(v_after)) * 3.6, 1),
                        "mismatch_pct": round(float(np.linalg.norm(v_b0 - v_after) / np.linalg.norm(v_b0)) * 100, 1)},
           "curves": {"t": t.tolist(), "cg_struck": np.round(cg_s, 4).tolist(), "cg_bullet_before": np.round(cg_b0, 4).tolist(),
                      "cg_bullet_after": np.round(cg_b1, 4).tolist(), "turn_struck": np.round(th_s, 5).tolist(),
                      "dvX": np.round(dvX, 4).tolist(), "dvY": np.round(dvY, 4).tolist()}}
    print(json.dumps({kk: vv for kk, vv in res.items() if kk != "curves"}, indent=1))
    if k1_override is None:
        (ROOT / "benchmark" / "yaris7371.json").write_text(json.dumps(res, default=float))
    return res


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else None)
