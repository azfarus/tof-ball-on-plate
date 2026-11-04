"""Build the MuJoCo scene: plate on 3 servo legs, rim, ball, corner sensors.

scan["corners"] lists the fitted corners (0..3). Sensor k sits at corner
scan["corners"][k]. With scan.actuator = "servo" each sensor body turns on a
servo (motor + backlash hinges about the plate normal); angle 0 points the
beam along the corner bisector.
"""
import numpy as np
import mujoco

from . import legs

# Corner signs in plate frame, and the inward bisector yaw of each corner (deg).
CORNERS = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
BISECTOR_DEG = [45.0, 135.0, 225.0, 315.0]


def corner_xy(plate, scan):
    h = plate["size_m"] / 2 - scan["mount_inset_m"]
    return [np.array([CORNERS[c][0] * h, CORNERS[c][1] * h]) for c in scan["corners"]]


def _rims(plate, fr):
    s, rh, rw = plate["size_m"] / 2, plate["rim_height_m"], plate["rim_width_m"]
    if rh <= 0:
        return ""  # open edge: the ball rolls off
    z = rh / 2
    boxes = [
        (0, s - rw / 2, s, rw / 2),
        (0, -(s - rw / 2), s, rw / 2),
        (s - rw / 2, 0, rw / 2, s),
        (-(s - rw / 2), 0, rw / 2, s),
    ]
    out = []
    for k, (x, y, hx, hy) in enumerate(boxes):
        out.append(
            f'<geom name="rim{k}" type="box" pos="{x} {y} {z}" '
            f'size="{hx} {hy} {rh / 2}" friction="{fr}" rgba=".2 .2 .2 1"/>'
        )
    return "\n        ".join(out)


def _sensors(plate, scan, pf, ss):
    """Sensor housings. One body per sensor so ray casts can skip their own housing."""
    r = scan["housing_radius_m"]
    hh = scan["mount_height_m"] + 0.008  # housing top sits above the beam axis
    out, acts = [], []
    for k, (xy, c) in enumerate(zip(corner_xy(plate, scan), scan["corners"])):
        mass = scan.get("rotor_mass_kg", 0.004)
        geom = (f'<geom name="housing{k}" type="cylinder" size="{r} {hh / 2}" '
                f'pos="0 0 {hh / 2}" mass="{mass}" rgba=".1 .3 .8 1"/>')
        if ss is None:
            out.append(f'<body name="sensor{k}" pos="{xy[0]} {xy[1]} 0">{geom}</body>')
            continue
        inner = lambda lash, k=k, geom=geom: f'<body name="sensor{k}">{lash}{geom}</body>'
        rng = ss["travel_deg"] / 2
        out.append(f'<body name="mount{k}" pos="{xy[0]} {xy[1]} 0" euler="0 0 {BISECTOR_DEG[c]}">'
                   f'{legs.servo_joint_xml(f"scan{k}", "0 0 1", rng, ss, pf["lash_solref"], inner)}'
                   '</body>')
        acts.append(f'<motor name="scan{k}" joint="scan{k}" ctrlrange="{-ss["t_stall"]} '
                    f'{ss["t_stall"]}" ctrllimited="true"/>')
    return "\n        ".join(out), acts


def build_xml(cfg, ls, ss=None):
    """MJCF string. ls, ss = leg and scan servo specs (ss None = ideal scan)."""
    plate, scan, pf = cfg["plate"], cfg["scan"], cfg["platform"]
    s, t = plate["size_m"] / 2, plate["thickness_m"]
    r, m = plate["ball_radius_m"], plate["ball_mass_kg"]
    inertia = (2 / 3 if plate["ball_hollow"] else 2 / 5) * m * r * r
    fr = " ".join(str(v) for v in plate["friction"])
    sensors, scan_acts = _sensors(plate, scan, pf, ss)
    leg_bodies, eqs, leg_acts = legs.legs_xml(pf, ls)
    acts = "\n    ".join(leg_acts + scan_acts)  # ctrl order: 3 legs, then scan servos
    return f"""<mujoco model="ball_on_plate_tof">
  <option timestep="{plate['timestep_s']}" integrator="implicitfast"
          cone="elliptic" impratio="10"/>
  <worldbody>
    <light pos="0 0 1"/>
    <geom name="base" type="cylinder" size="0.13 0.004" pos="0 0 0.004"
          contype="0" conaffinity="0" rgba=".35 .3 .25 1"/>
    <body name="plate" pos="0 0 {legs.neutral_height(pf, plate)}">
      <freejoint name="plate"/>
      <geom name="plate" type="box" size="{s} {s} {t / 2}" pos="0 0 {-t / 2}"
            mass="0.3" friction="{fr}" condim="6" rgba=".15 .15 .15 1"/>
      {_rims(plate, fr)}
      {sensors}
    </body>{leg_bodies}
    <body name="ball" pos="0 0 {r}">
      <freejoint name="ball"/>
      <inertial pos="0 0 0" mass="{m}" diaginertia="{inertia} {inertia} {inertia}"/>
      <geom name="ball" type="sphere" size="{r}" friction="{fr}" condim="6" priority="1"
            solref="{plate['ball_solref'][0]} {plate['ball_solref'][1]}" rgba="1 1 1 1"/>
    </body>
  </worldbody>
  <equality>
    {eqs}
  </equality>
  <actuator>
    {acts}
  </actuator>
</mujoco>"""


def build(cfg, ls, ss=None):
    """Return (model, data, ids). ids holds the indices the sim needs."""
    m = mujoco.MjModel.from_xml_string(build_xml(cfg, ls, ss))
    d = mujoco.MjData(m)
    name = lambda kind, n: mujoco.mj_name2id(m, kind, n)
    jnt = lambda n: name(mujoco.mjtObj.mjOBJ_JOINT, n)
    n = len(cfg["scan"]["corners"])
    jb = jnt("ball")
    ids = {
        "plate": name(mujoco.mjtObj.mjOBJ_BODY, "plate"),
        "ball": name(mujoco.mjtObj.mjOBJ_BODY, "ball"),
        "ball_qpos": m.jnt_qposadr[jb],
        "ball_dof": m.jnt_dofadr[jb],
        "plate_qpos": m.jnt_qposadr[jnt("plate")],
        "sensors": [name(mujoco.mjtObj.mjOBJ_BODY, f"sensor{k}") for k in range(n)],
        "housings": [name(mujoco.mjtObj.mjOBJ_GEOM, f"housing{k}") for k in range(n)],
        "albedo": albedo_table(m, cfg["plate"]["albedo"]),
    }
    for key, names in [("leg", [f"servo{i}" for i in range(3)]),
                       ("scan", [f"scan{k}" for k in range(n)] if ss else [])]:
        ids[key + "_qpos"] = np.array([m.jnt_qposadr[jnt(j)] for j in names], int)
        ids[key + "_lash"] = np.array([m.jnt_qposadr[jnt(j + "_lash")] for j in names], int)
        ids[key + "_dof"] = np.array([m.jnt_dofadr[jnt(j)] for j in names], int)
    ids["knee_qpos"] = np.array([m.jnt_qposadr[jnt(f"knee{i}")] for i in range(3)], int)
    return m, d, ids


def albedo_table(m, albedo):
    """Per-geom reflectance, looked up by geom name prefix."""
    table = np.zeros(m.ngeom)
    for g in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        for key, val in albedo.items():
            if n.startswith(key):
                table[g] = val
    return table


def place_ball(m, d, ids, plate, xy, vxy, z=None):
    """Put the ball at plate-frame xy with velocity vxy. z = centre height above
    the plate top (None = resting on the plate, rolling without slip)."""
    mujoco.mj_kinematics(m, d)
    R = d.xmat[ids["plate"]].reshape(3, 3)
    p0 = d.xpos[ids["plate"]]
    r = plate["ball_radius_m"]
    a = ids["ball_qpos"]
    d.qpos[a:a + 3] = p0 + R @ np.array([xy[0], xy[1], r + 1e-5 if z is None else z])
    d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
    v = ids["ball_dof"]
    vel = R @ np.array([vxy[0], vxy[1], 0.0])
    d.qvel[v:v + 3] = vel
    d.qvel[v + 3:v + 6] = np.cross(R[:, 2], vel) / r if z is None else 0.0
    mujoco.mj_forward(m, d)


def ball_plate_frame(d, ids):
    """Ball centre position and velocity in plate frame."""
    R = d.xmat[ids["plate"]].reshape(3, 3)
    p = R.T @ (d.xpos[ids["ball"]] - d.xpos[ids["plate"]])
    v = R.T @ d.qvel[ids["ball_dof"]:ids["ball_dof"] + 3]
    return p, v
