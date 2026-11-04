"""Scenario motion: plate tilt profiles and random forces on the ball.

Forces are unknown to the tracker. Kicks are short random pushes (Poisson in
time). The background push is a slowly varying random force (Ornstein-Uhlenbeck),
like table vibration or air.
"""
import numpy as np


def random_tilt(sc, rng):
    """Smooth random tilt: sum of sines per axis, weights sum to 1."""
    n = 6
    a = rng.uniform(0.5, 1.0, (3, n))  # rows: roll, pitch, heave
    return {
        "f": rng.uniform(0.1, sc["freq_hz"], (3, n)),
        "ph": rng.uniform(0, 2 * np.pi, (3, n)),
        "a": a / a.sum(1, keepdims=True),
    }


def tilt_at(sc, t, rp=None):
    """Plate tilt [about x, about y] in rad at time t."""
    t0 = np.radians(sc["tilt_deg"])
    t = max(t - sc.get("motion_delay_s", 0.0), 0.0)  # hold tilt_deg until the delay ends
    if sc["motion"] == "hold":
        return t0
    amp = np.radians(sc["amp_deg"])
    if sc["motion"] == "random":
        return t0 + amp * _ramp(t) * _rand_sum(rp, t)[:2]  # zero mean around tilt_deg
    w = 2 * np.pi * np.array(sc["freq_hz"]) * t
    if sc["motion"] == "sine":
        return t0 + amp * np.sin(w)
    if sc["motion"] == "circle":
        return np.array([amp[0] * np.cos(w[0]), amp[1] * np.sin(w[1])])
    raise ValueError(sc["motion"])


def _rand_sum(rp, t):
    return (rp["a"] * np.sin(2 * np.pi * rp["f"] * t + rp["ph"])).sum(1)


def _ramp(t, t_ramp=1.0):
    """Smooth 0 -> 1 over t_ramp seconds, so random motion starts gently."""
    x = min(t / t_ramp, 1.0)
    return x * x * (3 - 2 * x)


def heave_at(sc, t, rp=None):
    """Plate heave (m) at time t: random with "random" motion, else a sine."""
    amp = sc.get("heave_amp_m", 0.0)
    t = max(t - sc.get("motion_delay_s", 0.0), 0.0)
    if amp == 0.0:
        return 0.0
    if sc["motion"] == "random":
        return amp * _ramp(t) * _rand_sum(rp, t)[2]
    return amp * np.sin(2 * np.pi * sc.get("heave_freq_hz", 0.5) * t)


def kick_list(sc, rng, duration):
    """Random kicks: rows of [t_start, t_end, fx, fy] in plate frame (N)."""
    rate = sc.get("kicks_per_s", 0.0)
    n = rng.poisson(rate * duration)
    t = np.sort(rng.uniform(0.2, duration, n))
    ang = rng.uniform(0, 2 * np.pi, n)
    mag = sc.get("kick_force_n", 0.0) * rng.uniform(0.5, 1.5, n)
    dur = sc.get("kick_ms", 20.0) / 1000
    return np.stack([t, t + dur, mag * np.cos(ang), mag * np.sin(ang)], 1)


def force_at(sc, t, dt, kicks, push, rng):
    """Plate-frame xy force on the ball at time t. push is OU state, updated in place."""
    sigma, tau = sc.get("push_sigma_n", 0.0), sc.get("push_tau_s", 0.3)
    if sigma > 0:
        push += -push / tau * dt + sigma * np.sqrt(2 * dt / tau) * rng.normal(size=2)
    f = push.copy()
    for t0, t1, fx, fy in kicks:
        if t0 <= t < t1:
            f += [fx, fy]
    return f
