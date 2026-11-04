"""hybrid: a textured side plane filmed at an angle, moving along itself on a known decelerating curve."""
import cv2
import numpy as np

from physlens.hybrid import fit_side_plane, plane_point, track_part


def test_motion_on_an_oblique_plane():
    rng = np.random.default_rng(1)
    tex = cv2.GaussianBlur((rng.random((600, 2400)) * 255).astype(np.uint8), (0, 0), 1.4)   # 2 cm texels
    texel = 0.005
    w, h, f = 640, 400, 500.0
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    # The side plane: 6 m from the camera, turned 25° away; d along it, up = −y (OpenCV camera).
    a = np.radians(25)
    d = np.array([np.cos(a), 0.0, np.sin(a)])
    up = np.array([0.0, -1.0, 0.0])
    n = np.cross(d, up)
    c = np.array([0.0, 0.0, 6.0])
    ys, xs = np.mgrid[0:h, 0:w]
    px = np.c_[xs.ravel(), ys.ravel()].astype(float)
    X = plane_point(px, K, n, c)
    u0, v0 = (X - c) @ d, (X - c) @ up
    t = np.arange(40) / 1000
    x_true = 15.0 * t - 0.5 * 200.0 * t ** 2            # 54 km/h, decelerating at 20 g
    frames = []
    for x in x_true:
        mu = ((u0 - x) / texel + 1200).astype(np.float32).reshape(h, w)
        mv = (-v0 / texel + 300).astype(np.float32).reshape(h, w)
        frames.append(cv2.remap(tex, mu, mv, cv2.INTER_LINEAR))
    region = lambda i, sh: np.ones((h, w), bool)
    # The plane as the 3D world would give it: fitted to points on it.
    P = X[rng.choice(len(X), 3000, replace=False)] + rng.normal(0, 0.005, (3000, 3))
    n_fit, c_fit = fit_side_plane(P, d)
    r = track_part(frames, region, 0, len(frames) - 1, K, n_fit, c_fit, d)
    err = np.abs(r["along_m"] - x_true)
    assert err.max() < 0.01, err.max()                    # within a centimetre over the whole 40 ms


def test_track_rigid_turn_and_slide():
    """A textured patch on a plane 5 m from a straight-down camera turns 0.5° and slides 1 cm per frame: the
    tracker's pose gives both (image y points down, so a turn shown anticlockwise reads negative)."""
    import cv2
    from physlens.hybrid import track_rigid
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur((rng.random((600, 600)) * 255).astype(np.uint8), (5, 5), 0)
    K = np.array([[800, 0, 400], [0, 800, 300], [0, 0, 1.0]])
    frames = []
    for i in range(40):
        M = cv2.getRotationMatrix2D((300, 300), 0.5 * i, 1.0)
        M[0, 2] += 100 + 0.01 * i * 800 / 5.0
        frames.append(cv2.warpAffine(tex, M, (800, 600), borderValue=128))
    reg = np.zeros((600, 800), bool)
    reg[150:450, 250:550] = True
    r = track_rigid(frames, reg, list(range(40)), K, np.array([0, 0, -1.0]), np.array([0, 0, 5.0]),
                    np.array([1, 0, 0.]), np.array([0, 1, 0.]))
    th, t = r["theta"][-1], r["t"][-1]
    centre = np.array([(400 - 400) * 5 / 800, (300 - 300) * 5 / 800])       # the patch's centre at the seed frame
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    assert abs(np.degrees(th) + 19.5) < 0.2
    assert np.allclose(R @ centre + t - centre, [0.39, 0.0], atol=0.01)
