"""objsim: one body and its poses from partial views of a moving, turning box."""
import numpy as np

from physlens.objsim import build_body, coverage


def _box_surface(L=4.0, W=1.8, H=1.5, step=0.05, rng=None):
    pts = []
    xs, ys, zs = np.arange(-L / 2, L / 2, step), np.arange(0, H, step), np.arange(-W / 2, W / 2, step)
    for x in xs:
        for y in ys:
            pts += [[x, y, -W / 2], [x, y, W / 2]]
    for z in zs:
        for y in ys:
            pts += [[-L / 2, y, z], [L / 2, y, z]]
    for x in xs:
        for z in zs:
            pts.append([x, H, z])
    P = np.array(pts)
    return P + (rng.normal(0, 0.005, P.shape) if rng is not None else 0)


def test_poses_and_missing_side():
    rng = np.random.default_rng(0)
    body = _box_surface(rng=rng)
    cam = np.array([0.0, 3.0, 8.0])
    clouds, cols, centres, heads, truth = [], [], [], [], []
    for k in range(12):
        yaw = 0.02 * k                                   # turning slowly
        pos = np.array([-6 + 1.0 * k, 0.0, 0.3 * np.sin(k / 4)])
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
        W = body @ R.T + pos
        # The camera sees the side facing it (+z, the body's right) and the top: the far side (−z) never.
        normal_z = (body[:, 2] > 0.85) | (body[:, 1] > 1.45) | (np.abs(body[:, 0]) > 1.95)
        vis = normal_z & (W[:, 0] - pos[0] < 2.0 + (k % 3))     # plus a varying cut, as if partly out of view
        clouds.append(W[vis]); cols.append(np.full((vis.sum(), 3), 200, np.uint8))
        centres.append(np.median(W[vis], axis=0)); heads.append(yaw); truth.append((pos, yaw))
    b = build_body(clouds, cols, list(range(12)), [k / 10 for k in range(12)], 6, np.array(centres), np.array(heads))
    used = [p for p in b.poses if p["used"]]
    assert len(used) >= 10
    # Relative motion between frames matches the truth (the body origin is its own centre, so compare steps).
    for p, q in zip(used, used[1:]):
        k, j = p["frame"], q["frame"]
        dt = np.linalg.norm(truth[j][0][[0, 2]] - truth[k][0][[0, 2]])
        dm = np.hypot(q["x"] - p["x"], q["z"] - p["z"])
        assert abs(dm - dt) < 0.05
        assert abs((q["yaw"] - p["yaw"]) - (truth[j][1] - truth[k][1])) < np.radians(1.0)
    cov = coverage(b)
    assert abs(cov["size_m"]["length"] - 4.0) < 0.15
    # The camera sees +z, the body's right (x forward, y up); the left (−z) is never seen.
    assert "left" in cov["never_seen"] and "right" not in cov["never_seen"]
