"""Live MuJoCo viewer: beams coloured by last reading status, EKF estimate marker."""
import time

import numpy as np
import mujoco
import mujoco.viewer

from . import tof


def open_viewer(m, d, vcfg):
    v = mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False)
    v.cam.lookat[:] = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "plate")]
    v.cam.distance = vcfg["cam_distance_m"]
    v.cam.elevation = vcfg["cam_elevation_deg"]
    v.cam.azimuth = vcfg["cam_azimuth_deg"]
    return {"v": v, "cfg": vcfg, "t0": time.perf_counter() - d.time / vcfg["speed"], "next": 0.0}


def _line(scn, a, b, rgba, width):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                        np.eye(3).ravel(), np.array(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width, a, b)
    scn.ngeom += 1


def _sphere(scn, pos, r, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([r, 0, 0]), pos, np.eye(3).ravel(),
                        np.array(rgba, np.float32))
    scn.ngeom += 1


def draw(view, sim):
    """Draw overlays and keep playback at the configured speed. False = window closed."""
    v, c, d = view["v"], view["cfg"], sim.d
    if not v.is_running():
        return False
    t = d.time
    if t < view["next"]:
        return True
    view["next"] = t + 1.0 / c["fps"]
    col, L, w = c["colors"], c["beam_length_m"], c["line_width"]
    R = d.xmat[sim.ids["plate"]].reshape(3, 3)
    p0 = d.xpos[sim.ids["plate"]]
    on_plate = lambda xy: p0 + R @ np.array([xy[0], xy[1], sim.R_ball])
    with v.lock():
        scn = v.user_scn
        scn.ngeom = 0
        for i, (o, Rs) in enumerate(sim.sensor_poses()):
            rgba = col.get(sim.status[i], col["none"])
            half = tof.rx_half(sim.sn, L)
            _line(scn, o, o + L * Rs[:, 0], rgba, w)
            for a in (-half, half):
                _line(scn, o, o + L * (Rs @ np.array([np.cos(a), np.sin(a), 0.0])), rgba, w / 2)
            rd = sim.last[i]
            if rd is not None and rd.get("status") == "accepted":
                xy = sim.corners[i] + (rd["range"] + sim.R_ball) * np.array(
                    [np.cos(rd["yaw"]), np.sin(rd["yaw"])])
                _sphere(scn, on_plate(xy), 0.004, col["reading"])
        if sim.trk.ok:
            _sphere(scn, on_plate(sim.trk.x[:2]), sim.R_ball * 1.05, col["estimate"])
    v.sync()
    wait = view["t0"] + t / c["speed"] - time.perf_counter()
    if wait > 0:
        time.sleep(wait)
    return True
