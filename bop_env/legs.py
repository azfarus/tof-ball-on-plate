"""3-RRS plate support: kinematics and MJCF of the three servo legs.

Three servos on a base circle, 120 deg apart. Each servo turns a crank (horn)
in its vertical radial plane. A push rod hinges on the crank tip (same axis)
and meets the plate underside through a ball joint (MuJoCo connect equality).
The plate is a free body; the legs leave it roll, pitch and heave.

Leg frame: x radial outward, y tangential, z up, origin on the servo axis.
Servo angle q turns the crank tip to a (cos q, 0, -sin q); q < 0 lifts it.
q = 0 (centre pulse) is the crank horizontal and the plate flat.

Each servo is two hinges in series: the motor/gearbox joint (driven, with the
reflected rotor inertia) and the backlash joint (free between end stops).
The horn angle is their sum.
"""
import numpy as np


def leg_frames(pf):
    """(origin, rotation) of each leg frame in world."""
    out = []
    for a in np.radians(pf["leg_angles_deg"]):
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        B = np.array([pf["base_radius_m"] * np.cos(a), pf["base_radius_m"] * np.sin(a),
                      pf["servo_height_m"]])
        out.append((B, R))
    return out


def attach_points(pf, plate):
    """Ball joint points in plate frame (plate frame origin = top centre)."""
    r, t = pf["attach_radius_m"], plate["thickness_m"]
    return [np.array([r * np.cos(a), r * np.sin(a), -t]) for a in np.radians(pf["leg_angles_deg"])]


def neutral_height(pf, plate):
    """Plate top height with cranks horizontal and rods vertical."""
    return pf["servo_height_m"] + pf["rod_m"] + plate["thickness_m"]


def tilt_matrix(roll, pitch):
    """Plate rotation: about x (roll), then about y (pitch)."""
    cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return rx @ ry


def ik(pf, plate, roll, pitch, heave):
    """Servo and knee angles (rad) for a plate pose. Returns (q, knee)."""
    a, L = pf["crank_m"], pf["rod_m"]
    lim = np.radians(pf["range_deg"])
    centre = np.array([0, 0, neutral_height(pf, plate) + heave])
    R = tilt_matrix(roll, pitch)
    qs, knees = [], []
    for (B, Rl), p in zip(leg_frames(pf), attach_points(pf, plate)):
        P = Rl.T @ (centre + R @ p - B)  # ball joint in leg frame
        g = (P @ P + a * a - L * L) / (2 * a)
        n = np.hypot(P[0], P[2])
        base = np.arctan2(-P[2], P[0])
        acos = np.arccos(np.clip(g / n, -1, 1))  # clip = pose out of reach
        q = min([base + acos, base - acos], key=lambda x: abs((x + np.pi) % (2 * np.pi) - np.pi))
        q = (q + np.pi) % (2 * np.pi) - np.pi
        q = float(np.clip(q, -lim, lim))
        C = a * np.array([np.cos(q), 0, -np.sin(q)])
        knees.append(np.arctan2(P[0] - C[0], P[2] - C[2]) - q)
        qs.append(q)
    return np.array(qs), np.array(knees)


def pose(xmat, xpos, pf, plate):
    """Plate roll, pitch (rad) and heave (m) from its world pose."""
    R = xmat.reshape(3, 3)
    pitch = np.arcsin(np.clip(R[0, 2], -1, 1))
    roll = np.arctan2(-R[1, 2], R[2, 2])
    return roll, pitch, xpos[2] - neutral_height(pf, plate)


def servo_joint_xml(name, axis, rng_deg, s, lash_solref, inner):
    """Motor hinge + backlash hinge around `inner` (the horn body contents)."""
    b = s["backlash_deg"] / 2
    return f'''<body name="{name}_gear">
        <joint name="{name}" type="hinge" axis="{axis}" range="{-rng_deg} {rng_deg}"
               armature="{s['armature']}" frictionloss="{s['friction']}"/>
        <inertial pos="0 0 0" mass="0.001" diaginertia="1e-8 1e-8 1e-8"/>
        {inner(f'<joint name="{name}_lash" type="hinge" axis="{axis}" range="{-b} {b}" '
               f'solreflimit="{lash_solref[0]} {lash_solref[1]}" damping="1e-4" armature="1e-6"/>')}
      </body>'''


def legs_xml(pf, s):
    """Servo bodies (world children), plate-joint equalities and actuators."""
    a, L = pf["crank_m"], pf["rod_m"]
    rng = min(pf["range_deg"], s["travel_deg"] / 2)
    bodies, eqs, acts = [], [], []
    for i, ((B, _), ang) in enumerate(zip(leg_frames(pf), pf["leg_angles_deg"])):
        h = np.radians(ang) / 2
        horn = lambda lash, i=i: f'''<body name="crank{i}">
          {lash}
          <geom type="capsule" fromto="0 0 0 {a} 0 0" size="0.004" contype="0"
                conaffinity="0" rgba=".9 .9 .9 1" mass="0.005"/>
          <body name="rod{i}" pos="{a} 0 0">
            <joint name="knee{i}" type="hinge" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0 0 {L}" size="0.003" contype="0"
                  conaffinity="0" rgba=".7 .7 .2 1" mass="0.01"/>
          </body>
        </body>'''
        bodies.append(f'''
    <body name="servo{i}" pos="{B[0]} {B[1]} {B[2]}" quat="{np.cos(h)} 0 0 {np.sin(h)}">
      <geom name="servo_case{i}" type="box" size="0.02 0.01 0.02" pos="0 0.012 -0.01"
            contype="0" conaffinity="0" rgba=".2 .2 .25 1" mass="{s['mass_g'] / 1000}"/>
      {servo_joint_xml(f"servo{i}", "0 1 0", rng, s, pf["lash_solref"], horn)}
    </body>''')
        sr = pf["joint_solref"]
        eqs.append(f'<connect body1="rod{i}" body2="plate" anchor="0 0 {L}" '
                   f'solref="{sr[0]} {sr[1]}"/>')
        acts.append(f'<motor name="servo{i}" joint="servo{i}" ctrlrange="{-s["t_stall"]} '
                    f'{s["t_stall"]}" ctrllimited="true"/>')
    return "".join(bodies), "\n    ".join(eqs), acts
