"""VL53L4CD-style ToF model: cone ray casting, return histogram, interference.

Units: m, rad. Signal is relative received power. S_ref (reference_signal) is
the signal from a white 88% target filling the FoV at 200 mm.

Per sub-sample each sensor casts a fan of rays over its detection cone.
Each hit returns power ~ emission * albedo * cos(incidence) / (pi d^2).
Returns are binned by range; at the end of a reading the strongest peak gives
the range. Other sensors' light is unsynchronised, so it adds flat ambient
counts that raise noise (direct beam or diffuse bounce into the receiver).
"""
import numpy as np
import mujoco


def make_rays(sensor):
    """Equal-solid-angle ray fan over the widest detection cone."""
    n = sensor["n_rays"]
    half = np.radians(sensor["fov_full_deg_near"] / 2)
    i = np.arange(n) + 0.5
    cos_t = 1 - (1 - np.cos(half)) * i / n
    theta = np.arccos(cos_t)
    phi = i * np.pi * (3 - np.sqrt(5))  # golden-angle spiral
    sin_t = np.sin(theta)
    local = np.stack([cos_t, sin_t * np.cos(phi), sin_t * np.sin(phi)], axis=1)
    omega = 2 * np.pi * (1 - np.cos(half)) / n
    power = emission_weight(sensor, theta) * omega  # fraction of emitted power
    return {"local": local, "theta": theta, "power": power}


def emission_weight(sensor, theta):
    """VCSEL radiant intensity per sr (Gaussian, 1/e^2 half width), unit total."""
    w = np.radians(sensor["foi_half_deg"])
    return np.exp(-2 * theta**2 / w**2) / (np.pi * w**2 / 2)


def rx_half(sensor, dist):
    """Detection half-angle vs distance (DS13812 Table 16)."""
    near = np.radians(sensor["fov_full_deg_near"] / 2)
    far = np.radians(sensor["fov_full_deg_far"] / 2)
    return np.interp(dist, [sensor["fov_near_m"], sensor["fov_far_m"]], [near, far])


def sensor_frame(plate_R, plate_pos, xy, height, yaw, pitch):
    """World origin and rotation of a sensor. Local x is the beam axis."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
    origin = plate_pos + plate_R @ np.array([xy[0], xy[1], height])
    return origin, plate_R @ rz @ ry


def cast(m, d, origin, dirs, body_exclude, cutoff):
    """Cast rays from one point. Returns dist (-1 = miss), geom ids, normals."""
    n = len(dirs)
    geom = np.zeros(n, np.int32)
    dist = np.zeros(n)
    normal = np.zeros(3 * n)
    vec = np.ascontiguousarray(dirs, dtype=np.float64).ravel()
    mujoco.mj_multiRay(m, d, origin, vec, None, 1, body_exclude,
                       geom, dist, normal, n, cutoff)
    return dist, geom, normal.reshape(n, 3)


def emit(m, d, sensor, rays, origin, R, body):
    """Cast one sensor's beam. Keeps only rays that hit something."""
    dirs = rays["local"] @ R.T
    dist, geom, normal = cast(m, d, origin, dirs, body, sensor["max_range_m"])
    hit = dist > 0
    return {
        "origin": origin, "axis": R[:, 0], "body": body,
        "dirs": dirs[hit], "dist": dist[hit], "geom": geom[hit],
        "normal": normal[hit], "power": rays["power"][hit],
        "theta": rays["theta"][hit],
    }


def own_return(sensor, beam, albedo):
    """Range and power of each ray's echo back into the same sensor."""
    d = beam["dist"]
    ok = beam["theta"] <= rx_half(sensor, d)
    cos_in = np.clip(-(beam["normal"] * beam["dirs"]).sum(1), 0, 1)
    s = beam["power"] * albedo[beam["geom"]] * cos_in / (np.pi * d**2)
    return d[ok], s[ok]


def diffuse_interference(m, d, sensor, beam, rx_origin, rx_axis, rx_body, albedo):
    """Power of another sensor's spots that bounce into this receiver."""
    if len(beam["dist"]) == 0:
        return 0.0
    q = beam["origin"] + beam["dirs"] * beam["dist"][:, None]
    v = q - rx_origin
    di = np.linalg.norm(v, axis=1)
    u = v / di[:, None]
    theta = np.arccos(np.clip(u @ rx_axis, -1, 1))
    cos_out = -(beam["normal"] * u).sum(1)
    ok = (theta <= rx_half(sensor, di)) & (cos_out > 0)
    if not ok.any():
        return 0.0
    vis, _, _ = cast(m, d, rx_origin, u[ok], rx_body, sensor["max_range_m"] + 0.1)
    seen = np.abs(vis - di[ok]) < 1e-3  # nothing blocks the spot
    p = beam["power"][ok] * albedo[beam["geom"][ok]] * cos_out[ok] / (np.pi * di[ok] ** 2)
    return float(p[seen].sum())


def direct_interference(m, d, sensor, tx, tx_housing, rx_origin, rx_axis, rx_body):
    """Power of another sensor's VCSEL shining straight into this receiver."""
    v = tx["origin"] - rx_origin
    dist = np.linalg.norm(v)
    u = v / dist
    th_rx = np.arccos(np.clip(u @ rx_axis, -1, 1))
    th_tx = np.arccos(np.clip(-u @ tx["axis"], -1, 1))
    if th_rx > rx_half(sensor, dist) or th_tx > 3 * np.radians(sensor["foi_half_deg"]):
        return 0.0
    _, g, _ = cast(m, d, rx_origin, u[None, :], rx_body, dist + 0.1)
    if g[0] != tx_housing:
        return 0.0  # line of sight blocked (e.g. by the ball)
    return float(emission_weight(sensor, th_tx) / dist**2)


def new_acc(sensor):
    nbins = int(sensor["max_range_m"] / sensor["bin_m"]) + 1
    return {"hist": np.zeros(nbins), "amb": 0.0, "yaw": 0.0, "n": 0}


def sample_tick(m, d, sensor, rays, poses, ids, accs, interference):
    """One sub-sample for all sensors. poses = [(origin, R), ...]."""
    bodies, housings, albedo = ids["sensors"], ids["housings"], ids["albedo"]
    beams = [emit(m, d, sensor, rays, o, R, b) for (o, R), b in zip(poses, bodies)]
    for i, beam in enumerate(beams):
        dist, s = own_return(sensor, beam, albedo)
        idx = (dist / sensor["bin_m"]).astype(int)
        keep = idx < len(accs[i]["hist"])
        np.add.at(accs[i]["hist"], idx[keep], s[keep])
        accs[i]["n"] += 1
        if not interference:
            continue
        for j, other in enumerate(beams):
            if j == i:
                continue
            accs[i]["amb"] += diffuse_interference(
                m, d, sensor, other, beam["origin"], beam["axis"], bodies[i], albedo)
            accs[i]["amb"] += direct_interference(
                m, d, sensor, other, housings[j], beam["origin"], beam["axis"], bodies[i])


def _kernel(sensor):
    s = sensor["pulse_sigma_m"] / sensor["bin_m"]
    x = np.arange(-int(3 * s), int(3 * s) + 1)
    return np.exp(-0.5 * (x / s) ** 2)


def _pick_peak(sensor, sm):
    if sensor["peak_select"] == "nearest":
        top = sm.max()
        for k in range(1, len(sm) - 1):
            if sm[k] >= sensor["nearest_frac"] * top and sm[k] >= sm[k - 1] and sm[k] >= sm[k + 1]:
                return k
    return int(np.argmax(sm))


def finalize(sensor, acc, s_ref, offset=0.0, rng=None):
    """Turn an accumulated reading into range, sigma and a valid flag."""
    out = {"valid": False, "range": np.nan, "sigma": np.inf, "signal": 0.0, "ambient": 0.0}
    if acc["n"] == 0:
        return out
    h = acc["hist"] / acc["n"]
    amb = acc["amb"] / acc["n"] + sensor["ambient_frac"] * s_ref
    out["ambient"] = amb
    if h.sum() <= 0:
        return out
    pk = _pick_peak(sensor, np.convolve(h, _kernel(sensor), mode="same"))
    w = int(2 * sensor["pulse_sigma_m"] / sensor["bin_m"])
    lo, hi = max(pk - w, 0), min(pk + w + 1, len(h))
    sig = h[lo:hi].sum()
    if sig <= 0:
        return out
    centres = (np.arange(lo, hi) + 0.5) * sensor["bin_m"]
    mean_range = (h[lo:hi] * centres).sum() / sig
    amb_peak = amb * (hi - lo) * sensor["bin_m"] / sensor["ambient_span_m"]
    sigma = (sensor["sigma_ref_m"] * np.sqrt(s_ref / sig)
             * np.sqrt(sensor["ref_budget_ms"] / sensor["timing_budget_ms"])
             * np.sqrt(1 + amb_peak / sig))
    noise = rng.normal(0, sigma) if rng is not None else 0.0
    out.update(signal=sig, sigma=sigma, valid=bool(sigma <= sensor["sigma_threshold_m"]))
    if out["valid"]:
        out["range"] = mean_range + offset + noise
    return out


WALL_XML = """<mujoco><worldbody>
  <body name="sensor0" pos="0 0 0"><geom name="housing0" type="sphere" size="0.001"/></body>
  <geom name="wall" type="box" pos="{x} 0 0" size="0.001 2 2"/>
</worldbody></mujoco>"""


def wall_reading(sensor, rays, dist, albedo, s_ref, budget_ms=None, rng=None):
    """Reading of a flat wall filling the FoV at distance dist."""
    m = mujoco.MjModel.from_xml_string(WALL_XML.format(x=dist + 0.001))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    table = np.zeros(m.ngeom)
    table[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "wall")] = albedo
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sensor0")
    beam = emit(m, d, sensor, rays, np.zeros(3), np.eye(3), body)
    r, s = own_return(sensor, beam, table)
    acc = new_acc(sensor)
    np.add.at(acc["hist"], (r / sensor["bin_m"]).astype(int), s)
    acc["n"] = 1
    cfg = dict(sensor)
    if budget_ms:
        cfg["timing_budget_ms"] = budget_ms
    return finalize(cfg, acc, s_ref, rng=rng)


def reference_signal(sensor, rays):
    """S_ref: signal from a white 88% wall filling the FoV at 200 mm."""
    return wall_reading(sensor, rays, 0.2, 0.88, 1.0)["signal"]
