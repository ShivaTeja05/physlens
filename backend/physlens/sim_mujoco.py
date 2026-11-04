"""MuJoCo from an atomic scene: export the event as an MJCF model, simulate it, and fit the unknown physics until the
simulation reproduces the measured atoms (system identification).

Drop / bounce: a free sphere above a floor. Bounciness follows the law the engine found (it depends on impact
speed): e = e0 − e1·v, applied at each impact through the contact's damping ratio
(e = exp(−ζπ/√(1−ζ²)) for MuJoCo's spring-damper contact). Fitted: gravity, e0, e1 and the start state.

Crash: the car on a slide joint along the track, a fixed barrier. MuJoCo's soft contact between them plays the
crumple zone: its time constant (stiffness) and damping ratio decide how far the car's rigid rear travels after
impact (the crush) and how fast it rebounds. After the rebound a drag force (wheels and wreckage dragging, μ·m·g)
stops it. Fitted: contact time constant and damping, drag μ and the start state, against the car's measured rear
edge. MuJoCo's axes are z up; the scene's are y up, and the mapping is done here.

The fitted model is written out as MJCF, so the same event opens in any MuJoCo viewer.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.optimize import least_squares


def _zeta(e: float) -> float:
    """Damping ratio of a spring-damper contact that gives restitution e."""
    e = min(max(e, 1e-4), 0.999)
    le = math.log(e)
    return -le / math.sqrt(math.pi ** 2 + le * le)


# --- drop -----------------------------------------------------------------------------------------------------

CONTACT_S = 0.005                        # a ball's contact with the floor lasts a few ms (a longer contact delays every bounce)
K_CONTACT = (math.pi / CONTACT_S) ** 2   # spring stiffness per unit mass for that contact time


def _damping(e: float) -> float:
    return 2 * _zeta(e) * math.sqrt(K_CONTACT)


def drop_mjcf(r: float, g: float, x0: float, y0: float, dt: float = 0.00025, e: float = 0.7,
              vx0: float = 0.0, vy0: float = 0.0) -> str:
    # Direct spring-damper contact (negative solref): MuJoCo's default soft contact pumps energy into bouncy
    # collisions (a target bounciness of 0.9 came back as 20), the direct form keeps it within a few percent.
    return f"""<mujoco model="physlens-drop">
  <option timestep="{dt}" gravity="0 0 {-g:.5f}"/>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" contype="1" conaffinity="1"/>
    <body name="ball" pos="{x0:.5f} 0 {y0:.5f}">
      <freejoint/>
      <geom name="ball" type="sphere" size="{r:.5f}" mass="0.1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <contact>
    <pair geom1="floor" geom2="ball" condim="1" solref="{-K_CONTACT:.1f} {-_damping(e):.3f}" solimp="0.9999 0.9999 0.001"/>
  </contact>
  <keyframe>
    <key name="measured_start" qpos="{x0:.5f} 0 {y0:.5f} 1 0 0 0" qvel="{vx0:.5f} 0 {vy0:.5f} 0 0 0"/>
  </keyframe>
</mujoco>"""


def simulate_drop(r: float, p: dict, t0: float, t1: float, dt: float = 0.00025):
    import mujoco
    m = mujoco.MjModel.from_xml_string(drop_mjcf(r, p["g"], p["x0"], p["y0"], dt))
    d = mujoco.MjData(m)
    d.qvel[0], d.qvel[2] = p["vx0"], p["vy0"]
    n = int((t1 - t0) / dt) + 1
    out = np.empty((n, 3))
    for i in range(n):
        vz = d.qvel[2]
        if vz < 0:   # falling: the next contact gets the bounciness for this impact speed
            e = min(max(p["e0"] - p["e1"] * (-vz), 0.05), 0.98)
            m.pair_solref[0, 1] = -_damping(e)
        out[i] = (t0 + i * dt, d.qpos[0], d.qpos[2])
        mujoco.mj_step(m, d)
    return out


def fit_drop(scene: dict) -> dict:
    obj = scene["objects"][0]
    r = obj["size"]["value"][0]
    atoms = [a for a in scene["atoms"][obj["id"]]]
    a0 = next((a for a in atoms if math.hypot(*a["v"][:2]) > 0.1), atoms[0])
    A = [a for a in atoms if a["t"] >= a0["t"]]
    T = np.array([a["t"] for a in A]); X = np.array([a["p"][0] for a in A]); Y = np.array([a["p"][1] for a in A])
    g0 = abs(scene["world"]["gravity"][1]); e_med = obj["restitution"]["value"]
    # Gravity stays at the engine's measurement (from the free-fall arcs, far more precise than a whole-clip fit,
    # which would bend it to absorb contact-model details); MuJoCo fits the contact law and the start state.
    names = ["e0", "e1", "y0", "vy0", "x0", "vx0"]
    start = np.array([e_med, 0.0, a0["p"][1], a0["v"][1], a0["p"][0], a0["v"][0]])
    lo = np.array([0.1, -0.1, a0["p"][1] - 0.3, a0["v"][1] - 3, a0["p"][0] - 0.3, a0["v"][0] - 1])
    hi = np.array([0.99, 0.2, a0["p"][1] + 0.3, a0["v"][1] + 3, a0["p"][0] + 0.3, a0["v"][0] + 1])

    def run(v):
        return simulate_drop(r, {"g": g0, **dict(zip(names, v))}, T[0], T[-1] + 0.5)

    def resid(v):
        s = run(v)
        return np.r_[np.interp(T, s[:, 0], s[:, 2]) - Y, 0.3 * (np.interp(T, s[:, 0], s[:, 1]) - X)]

    before = float(np.sqrt(np.mean(resid(start)[: len(T)] ** 2)))
    # A stepped simulation responds to its parameters in small jumps, so a derivative-free search (Powell) is more
    # reliable here than finite-difference gradients (which stalled at the starting point on the real clip).
    from scipy.optimize import minimize
    best = minimize(lambda v: float(np.sum(resid(v) ** 2)), start, method="Powell", bounds=list(zip(lo, hi)),
                    options={"maxfev": 1500, "xtol": 1e-4, "ftol": 1e-7})
    p = {"g": g0, **dict(zip(names, best.x))}
    s = run(best.x)
    rms = float(np.sqrt(np.mean((np.interp(T, s[:, 0], s[:, 2]) - Y) ** 2)))
    return {"kind": "drop", "params": {k: round(float(v), 4) for k, v in p.items()},
            "meaning": {"g": f"gravity, m/s² ({scene['world']['gravity_source']})", "e0, e1": f"bounciness = e0 − e1·impact speed ({p['e0']:.3f} − {p['e1']:.4f}·v)"},
            "rms_m": rms, "rms_unfitted_m": before,
            "track": [[round(float(t), 4), round(float(x), 4), round(float(y), 4), 0.0] for t, x, y in s[::4]],
            "mjcf": drop_mjcf(r, p["g"], p["x0"], p["y0"], e=p["e0"], vx0=p["vx0"], vy0=p["vy0"])}


# --- crash ----------------------------------------------------------------------------------------------------

def crash_mjcf(L: float, h: float, w: float, mass: float, x0: float, tc: float, zeta: float, dt: float = 0.0005,
               vx0: float = 0.0) -> str:
    return f"""<mujoco model="physlens-crash">
  <option timestep="{dt}" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="barrier" type="box" size="0.3 1.5 1.0" pos="-0.3 0 1.0" contype="0" conaffinity="0"/>
    <body name="car" pos="{x0:.5f} 0 {h / 2:.4f}">
      <joint name="track" type="slide" axis="1 0 0"/>
      <geom name="car" type="box" size="{L / 2:.4f} {w / 2:.4f} {h / 2:.4f}" mass="{mass:.1f}" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <contact>
    <pair geom1="barrier" geom2="car" condim="1" solref="{tc:.5f} {zeta:.4f}" solimp="0.95 0.99 0.001 0.5 2"/>
  </contact>
  <keyframe>
    <key name="measured_start" qpos="0" qvel="{vx0:.4f}"/>
  </keyframe>
</mujoco>"""


def simulate_crash(L, h, w, mass, p: dict, t0: float, t1: float, dt: float = 0.0005):
    import mujoco
    m = mujoco.MjModel.from_xml_string(crash_mjcf(L, h, w, mass, p["x0"], p["tc"], p["zeta"], dt))
    d = mujoco.MjData(m)
    d.qvel[0] = p["vx0"]
    n = int((t1 - t0) / dt) + 1
    out = np.empty((n, 3))
    hit = False
    for i in range(n):
        x, v = d.qpos[0] + p["x0"], d.qvel[0]
        hit = hit or (x - L / 2) < 0.0
        # After the impact, once moving back off the barrier, wheels and wreckage drag it to a stop.
        d.qfrc_applied[0] = -math.copysign(p["mu"] * mass * 9.81, v) if hit and v > 0.02 else 0.0
        if hit and 0 <= v <= 0.02 and (x - L / 2) > 0:
            d.qvel[0] = 0.0
        out[i] = (t0 + i * dt, x, L / 2 + x)          # centre, rigid rear
        mujoco.mj_step(m, d)
    return out


def fit_crash(scene: dict) -> dict:
    obj = scene["objects"][0]
    L, h, w = obj["size"]["value"]
    mass = obj["mass_kg"]["value"] or 1500.0
    atoms = scene["atoms"][obj["id"]]
    T = np.array([a["t"] for a in atoms])
    rear = np.array([a["p"][0] + a.get("length", L) / 2 for a in atoms])
    a0 = atoms[0]
    i_fit = np.nonzero(np.abs([a["v"][0] for a in atoms]) > 1.0)[0]
    a0 = atoms[int(i_fit[0])] if len(i_fit) else a0
    keep = T >= a0["t"]
    T, rear = T[keep], rear[keep]
    mu0 = obj.get("post_impact_friction", {}).get("value", 0.4)
    names = ["tc", "zeta", "mu", "x0", "vx0"]
    start = np.array([0.04, 0.3, mu0, a0["p"][0], a0["v"][0]])
    lo = np.array([0.005, 0.02, 0.0, a0["p"][0] - 0.5, a0["v"][0] - 3])
    hi = np.array([0.2, 1.5, 1.5, a0["p"][0] + 0.5, a0["v"][0] + 3])

    def run(v):
        return simulate_crash(L, h, w, mass, dict(zip(names, v)), T[0], T[-1] + 0.2)

    def resid(v):
        s = run(v)
        return np.interp(T, s[:, 0], s[:, 2]) - rear

    before = float(np.sqrt(np.mean(resid(start) ** 2)))
    fit = least_squares(resid, start, bounds=(lo, hi), diff_step=1e-3, max_nfev=150)
    p = dict(zip(names, fit.x))
    s = run(fit.x)
    rms = float(np.sqrt(np.mean((np.interp(T, s[:, 0], s[:, 2]) - rear) ** 2)))
    # What the fitted contact means physically: penetration at its deepest = crush; outgoing ÷ incoming = restitution.
    pen = float(max(0.0, -(s[:, 1] - L / 2).min()))
    v_in = abs(fit.x[4])
    v_out = float(np.max(np.diff(s[:, 1]) / np.diff(s[:, 0])))
    return {"kind": "crash", "params": {k: round(float(v), 4) for k, v in p.items()},
            "meaning": {"tc": "contact time constant, s (crumple-zone stiffness)", "zeta": "contact damping ratio",
                        "mu": "drag after rebound (μ)", "crush_m": round(pen, 3),
                        "restitution": round(v_out / v_in, 3) if v_in else None, "impact_speed_kmh": round(v_in * 3.6, 2)},
            "rms_m": rms, "rms_unfitted_m": before,
            "track": [[round(float(t), 4), round(float(x), 4), round(h / 2, 4), 0.0] for t, x, _ in s[::8]],
            "mjcf": crash_mjcf(L, h, w, mass, p["x0"], p["tc"], p["zeta"], vx0=p["vx0"])}


def fit(scene: dict) -> dict:
    return fit_crash(scene) if scene["objects"] and scene["objects"][0]["shape"] == "box" else fit_drop(scene)
