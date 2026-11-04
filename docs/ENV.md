# BallOnPlateToF-v0 environment

`BallOnPlateToF-v0` is a Gymnasium environment for balancing a 40 mm ball on a 30 cm plate. Three hobby PWM servos tilt the plate, and four VL53L4CD time-of-flight sensors scanning from the corners track the ball. The physics runs in MuJoCo. The package is `bop_env/`, every parameter is in `config/*.toml`, and the environment code knows nothing about learning (controllers and training live in `control/`).

```python
import gymnasium as gym, bop_env

env = gym.make("BallOnPlateToF-v0", overrides={"platform.servo_model": "DS3218"})
obs, info = env.reset(seed=0)
obs, r, terminated, truncated, info = env.step(env.action_space.sample())

venv = bop_env.make_vec(14)    # 14 headless copies, one process each
```

Contents:
1. [Package layout](#1-package-layout)
2. [Physical system](#2-physical-system)
3. [Action](#3-action)
4. [Observation](#4-observation)
5. [Reward, termination, info](#5-reward-termination-info)
6. [Reset and randomisation](#6-reset-and-randomisation)
7. [Configuration and overrides](#7-configuration-and-overrides)
8. [Servo model](#8-servo-model)
9. [Platform kinematics](#9-platform-kinematics)
10. [Scan and ToF sensor model](#10-scan-and-tof-sensor-model)
11. [EKF tracker](#11-ekf-tracker)
12. [Disturbances](#12-disturbances)
13. [Timing of one step](#13-timing-of-one-step)
14. [Parallelism and hardware acceleration](#14-parallelism-and-hardware-acceleration)
15. [Using Sim directly](#15-using-sim-directly)
16. [Known limits](#16-known-limits)

---

## 1. Package layout

| File | Role |
|---|---|
| `bop_env/__init__.py` | Registers `BallOnPlateToF-v0`. Exports `BallOnPlateEnv`, `Sim`, `load_all`, `make_vec`. |
| `bop_env/env.py` | `BallOnPlateEnv` (Gymnasium API) and `make_vec(n)`. |
| `bop_env/sim.py` | `Sim`: the one physics loop, shared by the env and `tools/`. |
| `bop_env/model.py` | Builds the MJCF: plate, ball, optional rim, sensor housings, scan servos. |
| `bop_env/legs.py` | 3-RRS geometry, closed-form IK, plate pose readout, MJCF for the servo legs. |
| `bop_env/servo.py` | Hobby PWM servo model (`spec`, `ServoBank`). |
| `bop_env/scan.py` | Triangle-wave scan schedule per corner sensor. |
| `bop_env/tof.py` | Ray-cast VL53L4CD model: histogram, peak, sigma, validity, interference. |
| `bop_env/tracker.py` | EKF on the plate-frame ball state. |
| `bop_env/disturb.py` | Random ball forces (kicks plus slow push) and the scripted tilt profiles used by `tools/`. |
| `bop_env/config.py` | Loads every `config/*.toml` and applies overrides. |
| `bop_env/viewer.py` | Passive MuJoCo viewer (`render_mode="human"`). |

| Config file | Contents |
|---|---|
| `config/env.toml` | Env only: control rate, episode length, observation options, reward, reset ranges, disturbances, env-only overrides, randomisation. |
| `config/plate.toml` | Plate size, ball, friction, contact, physics timestep, albedo. |
| `config/platform.toml` | 3-RRS geometry, plate servo model, supply voltage, PWM generator. |
| `config/servos.toml` | Servo library: every positional servo on the roboticsbd listing. |
| `config/scan.toml` | Fitted corners, scan pattern, mount, scan actuator and its servo. |
| `config/sensor.toml` | VL53L4CD: timing budget, FoV, ray model, noise, validity. |
| `config/tracker.toml` | EKF tuning. |
| `config/scenarios.toml`, `ablation.toml`, `viewer.toml` | Used by `tools/`, not by the env (except the viewer). |
| `config/control.toml` | PID and RL hyperparameters. Read by `control/` only, never by the env. |

---

## 2. Physical system

All coordinates are in the **plate frame**: origin at the plate top centre, x and y along the plate edges, z up. Units are SI (m, s, rad).

- **Plate:** 0.30 m square, 6 mm thick, matte black (albedo 0.10 at 940 nm). It is a free body carried by the three legs.
- **Ball:** 40 mm diameter, 2.7 g, hollow shell (I = ⅔mr²), white (albedo 0.85). Friction: slide 0.6, torsion 0.005, rolling 3e-5 m. The contact `ball_solref` gives 0.85 restitution.
- **Edge:** the env has **no rim** (`env.overrides` sets `plate.rim_height_m = 0`), so the ball can roll off. The scripted `tools/` runs keep the 25 mm rim.
- **Platform:** three servo cranks at 90°, 210° and 330°, with push rods to ball joints under the plate (3-RRS, see §9).
- **Sensors:** one VL53L4CD per fitted corner (`scan.corners`, default all four). Corner index: 0 (−,−), 1 (+,−), 2 (+,+), 3 (−,+). The beam sits 20 mm above the plate top, at the ball's equator, and the sensors tilt with the plate. Each sensor sits on a scan servo (§10).
- **Physics:** 4 kHz (`plate.timestep_s` = 0.25 ms). Gravity acts along −z of the world frame.

---

## 3. Action

`Box(-1, 1, (3,), float32)`: the target angles of the three plate servos, each divided by `platform.range_deg` (80°).

- 0 means the centre pulse (1500 µs), which is crank horizontal and **plate flat** at neutral height.
- ±1 means ±80° crank angle. The pulse is clipped to the servo's 500–2500 µs range (MG996R: ±90°).
- Out-of-range actions are clipped to [−1, 1].
- The servo latches a new pulse only once per PWM frame (`platform.servo_update_hz`, 50 Hz, capped by the model's `max_frame_hz`). Running `control_hz` above the frame rate has no effect.
- The action order follows `platform.leg_angles_deg`: leg 0 at 90°, leg 1 at 210°, leg 2 at 330°.
- Speed is not an input, as on the real servo. The optional host-side speed limit `platform.trajectory_speed_rps` is the only speed control.

To command a plate tilt instead, use the IK: `q, _ = bop_env.legs.ik(cfg["platform"], cfg["plate"], roll, pitch, heave)` gives servo angles in rad; divide by `radians(range_deg)`. `control/pid.py` does this.

---

## 4. Observation

A `Dict` of `float32` boxes, in SI units.

| Key | Shape | Contents |
|---|---|---|
| `ball_est` | (5,) | EKF estimate `[x, y, vx, vy, tracked]`, plate frame. `tracked` is 1 when the EKF holds a fix, else 0 (the state is then stale or zero). |
| `tof` | (n_sensors, tof_history, 7) | Raw ToF readings, newest first per sensor. Fields below. |
| `servo_cmd` | (3,) | Last action (the clipped one). |
| `target` | (2,) | Goal ball position (m). |
| `servo_fb` | (3,) | Optional (`env.servo_feedback = true`). Horn angles (rad) read from the pot wire, with `feedback_noise_deg` noise. Needs a feedback servo model (`MG996R_FB`); otherwise the constructor raises `ValueError`. |
| `plate` | (3,) | Optional (`env.plate_pose = true`). Measured plate `[roll, pitch, heave]` (rad, rad, m), as from an IMU plus height. Noise-free. |

ToF fields (index → name in `bop_env.env.TOF_FIELDS`):

| # | Field | Meaning |
|---|---|---|
| 0 | `valid` | 1 if the sensor reported a valid range (sigma ≤ `sigma_threshold_m`). |
| 1 | `range_m` | Measured range (m) to the strongest return, with noise and unit offset. 0 when invalid. Can be the ball, the plate edge or background. |
| 2 | `sigma_m` | Sensor-reported 1σ range noise (m). 0 when invalid. |
| 3 | `signal` | Return signal / S_ref (S_ref = white 88% wall at 200 mm). |
| 4 | `ambient` | Ambient / S_ref. Comes from interference by the other sensors' 940 nm light. |
| 5 | `yaw_rad` | Plate-frame beam bearing during the reading, as the host knows it: the measured scan-horn angle plus feedback noise (`scan.yaw_source = "measured"`), or the commanded yaw (`"commanded"`). |
| 6 | `age_s` | Time since the reading finished (s). |

Notes:
- Each sensor gives one reading per 10 ms timing budget, so a 20 ms step brings about two new readings per sensor (`tof_history = 2` by default).
- Slots never filled since reset are all zeros, including `age_s` until the first reading arrives.
- Raw readings are not background-filtered: the EKF filters them (§11), the observation does not. A policy can learn from `tof` alone, from `ball_est` alone, or both.
- The bounds are ±inf. Use observation normalisation (`VecNormalize` in `control/train.py`).

---

## 5. Reward, termination, info

Per step:

```
r = 1 − w_pos · |p − target| / half − w_vel · |v| − w_rate · |Δa|² − w_act · |a|²
```

- `p`, `v`: ball position and velocity in the plate frame. They come from the true state by default (`env.reward_source = "truth"`), or from the EKF (`"estimate"`).
- `half` = half the plate size (0.15 m), so `w_pos` is per half-plate of error.
- `Δa` = change in action since the last step, `a` = action.
- Defaults (`[reward]`): `w_pos` 1, `w_vel` 0.1, `w_rate` 0.05, `w_act` 0. A ball held at the target with a still plate earns about 1 per step (500 per 10 s episode).

**Termination (`terminated = True`):** the ball falls off. This is checked every physics step: the ball centre passes the plate edge (|x| or |y| > half), or drops below the plate top. The step stops early, returns `−escape_penalty` (20) and ends the episode.

**Truncation (`truncated = True`):** `episode_s` (10 s) reached, i.e. 500 steps at 50 Hz.

`info` dict:

| Key | Meaning |
|---|---|
| `ball_xy`, `ball_v` | True ball position and velocity (plate frame). |
| `est_xy` | EKF position. |
| `tracked` | EKF has a fix. |
| `err_m` | True distance to the target. |
| `escaped` | Ball fell off. |
| `plate_pose` | True `[roll, pitch, heave]`. |
| `servo_saturated` | Per plate servo: the motor was at full drive at the last physics step. |
| `t` | Sim time (s). |

---

## 6. Reset and randomisation

`reset(seed, options)` puts the plate flat with the servos at 0, places the ball, clears the EKF and draws new disturbances. Defaults are in `[start]`, each drawn uniform in ±value:
- ball position ±0.08 m per axis;
- ball velocity ±0.10 m/s per axis;
- target ±`target_xy_m` (0 means always the centre);
- `drop_height_m` > 0 releases the ball from up to that height above the plate.

`options` keys (all optional):

| Key | Effect |
|---|---|
| `ball_xy`, `ball_v` | Fix the start position or velocity (2-vectors). |
| `target` | Fix the goal. |
| `overrides` | Config overrides for this episode and later ones; the model is rebuilt. |

**Domain randomisation:** put ranges in `config/env.toml [randomize]` (or pass `"env.randomize": {...}`):

```toml
[randomize]
"servos.MG996R.backlash_deg" = [0.2, 1.5]
"sensor.offset_sigma_m" = [0.001, 0.004]
"platform.supply_v" = [5.0, 6.6]
```

Each key is drawn uniform in `[lo, hi]` at every reset and the MuJoCo model is rebuilt. A rebuild also recomputes the empty-plate background map, so it costs about 0.35 s (a plain reset is ~0). Only scalar values can be randomised; list items (like one friction component) cannot.

Per-episode random parts that need no rebuild: ToF noise, per-sensor range offset (`sensor.offset_sigma_m`), kick times and sizes, the slow push, the start state.

---

## 7. Configuration and overrides

`BallOnPlateEnv(overrides=None, render_mode=None, config_dir=config.CONFIG_DIR)`

Config is built in this order:
1. every `config/*.toml`, each under its file name (`plate`, `sensor`, `scan`, `tracker`, `scenarios`, `platform`, `servos`, `env`);
2. `env.overrides` (removes the rim for the env);
3. the caller's `overrides`.

Overrides take dotted keys at any depth, or nested dicts:

```python
BallOnPlateEnv(overrides={
    "platform.servo_model": "DS3218",
    "platform.servo_update_hz": 333,
    "servos.DS3218.deadband_us": 2,
    "scan.actuator": "ideal",
    "scan.corners": [0, 1],
    "env.control_hz": 100,
    "env.reward": {"w_vel": 0.2},
    "plate.rim_height_m": 0.025,       # put the rim back (ball can no longer fall off)
})
```

From the command line, the `tools/` and `control/` scripts accept `--set section.key=value` with a JSON value, e.g. `--set scan.corners=[0,2]`.

Main env keys (`config/env.toml`):

| Key | Default | Meaning |
|---|---|---|
| `control_hz` | 50 | Steps per second; one step = `1/control_hz` of physics. |
| `episode_s` | 10 | Truncation time. |
| `tof_history` | 2 | Readings kept per sensor in `obs["tof"]`. |
| `servo_feedback` | false | Add `servo_fb` (needs `MG996R_FB`). |
| `plate_pose` | false | Add `plate`. |
| `reward_source` | `"truth"` | `"truth"` or `"estimate"`. |
| `interference` | true | Model cross-talk between sensors. |
| `[start]`, `[reward]`, `[disturb]` | see §5, §6, §12 | |
| `[overrides]` | rim off | Applied to the other configs for the env only. |
| `[randomize]` | empty | Per-reset ranges. |

---

## 8. Servo model

`bop_env/servo.py` and `config/servos.toml` cover both the plate servos and the scan servos. Every model takes 500–2500 µs pulses with a 1500 µs centre. Select one with `platform.servo_model` (plate) or `scan.servo_model` (scan), and set its supply with `supply_v`.

### Host side (the controller's PWM generator)
1. The angle becomes a pulse: `pulse = 1500 + q · (2000 µs / travel)`.
2. It is quantised to `platform.pwm_resolution_us` (1 µs; a PCA9685 at 50 Hz is about 4.9 µs) and clipped to 500–2500 µs.
3. It is latched once per frame at `min(servo_update_hz, max_frame_hz)`: 50 Hz analog servos, up to 333 Hz for the DS3218.
4. An optional speed limit, `trajectory_speed_rps`, moves the latched pulse at most that fast.

### Servo side (inside the servo)
- The pot sits on the horn, after the gear backlash. Pulse error `e` (µs) = latched pulse − horn angle in µs.
- Deadband: `|e| ≤ deadband_us/2` gives no drive.
- P loop: `duty = clip(e / prop_band, −1, 1)`, where `prop_band_deg` is the error that gives full drive.
- DC motor with a linear torque-speed curve: `τ = T_stall · (duty − ω/ω_noload)`. The horn tops out at the rated no-load speed and stalls at the rated torque.
- Speed and torque are interpolated at `supply_v` between the listed voltages (clamped to the listed range).

### In MuJoCo
Each servo is two hinges in series:
- a **motor hinge** carrying the actuator, with reflected armature `mech_time_const_s · T_stall / ω_noload` and gear friction `friction_frac · T_stall`;
- a **backlash hinge**, free within ±`backlash_deg/2` with stiff end stops (`platform.lash_solref`).

Horn angle = motor + backlash. The motor is driven by a `motor` actuator with ctrlrange ±T_stall. The ctrl order is the 3 plate servos, then the scan servos.

### Parameters per model (`[default]` fills anything a model omits)

| Key | Meaning | Default |
|---|---|---|
| `kind` | `"analog"` / `"digital"` | |
| `travel_deg` | Angle across 500–2500 µs | |
| `volts`, `sec_per_60`, `torque_kgcm` | Rated speed and stall torque at each listed voltage | |
| `mass_g` | Mass | |
| `max_frame_hz` | Fastest PWM frame accepted | 50 |
| `deadband_us` | Ignored pulse change | 5 |
| `prop_band_deg` | P band (assumed) | 4 |
| `backlash_deg` | Gear play, total (assumed) | 0.5 |
| `mech_time_const_s` | Time to ~63% of top speed (assumed) | 0.02 |
| `friction_frac` | Gear friction / stall torque (assumed) | 0.02 |
| `feedback` | `"none"` or `"pot"` | none |
| `feedback_noise_deg` | Pot noise, 1σ (assumed) | 0.5 |
| `assumed` | Keys of this model not on the listing | |

### Library (listed values; `assumed` = filled in, not on the listing)

| Model | Kind | Travel° | Volts | s/60° | kg·cm | Frame Hz | Deadband µs | Feedback | Assumed |
|---|---|---|---|---|---|---|---|---|---|
| SG90 | analog | 180 | 4.8, 6.6 | 0.10, 0.10 | 1.2, 1.6 | 50 | 10 | – | deadband, speed @6.6 |
| SG92R | analog | 180 | 4.8 | 0.10 | 2.5 | 50 | 10 | – | speed, deadband |
| MG995 | analog | 180 | 4.8, 6.6 | 0.20, 0.16 | 10, 12 | 50 | 1 | – | |
| **MG996R** (plate default) | analog | 180 | 4.8, 6.6 | 0.19, 0.15 | 9.4, 11 | 50 | 1 | – | |
| MG996R_270 | analog | 270 | 4.8, 6.6 | 0.19, 0.15 | 9.4, 11 | 50 | 1 | – | |
| MG996R_FB | analog | 180 | 4.8, 6.6 | 0.19, 0.15 | 9.4, 11 | 50 | 1 | pot | |
| MG945 | digital | 180 | 4.8, 6.0 | 0.17, 0.14 | 10, 12 | 50 | 1 | – | speed, mass |
| DS3218 | digital | 270 | 5.0, 6.8 | 0.16, 0.14 | 19, 21.5 | 333 | 3 | – | |
| TD8120MG | digital | 180 | 4.8, 7.2 | 0.18, 0.14 | 20.5, 22.8 | 50 | 5 | – | |
| TD8120MG_270 | digital | 270 | 4.8, 7.2 | 0.18, 0.14 | 20.5, 22.8 | 50 | 5 | – | |
| TD8135MG | digital | 180 | 4.8, 8.4 | 0.32, 0.22 | 32.7, 35.2 | 50 | 5 | – | deadband |
| TD8135MG_270 | digital | 270 | 4.8, 8.4 | 0.32, 0.22 | 32.7, 35.2 | 50 | 5 | – | deadband |
| LDX218 | digital | 180 | 6.0, 7.4 | 0.16, 0.16 | 15, 17 | 50 | 3.3 | – | speed @6.0 |
| LDX227 | digital | 270 | 6.0, 7.4 | 0.16, 0.16 | 15, 17 | 50 | 3 | – | speed @6.0 |
| MAG25 | digital | 360 | 6.0, 7.4 | 0.185, 0.151 | 20.5, 25 | 50 | 1 | – | mass, deadband |
| DFR_CLUTCH_9G | analog | 180 | 4.8, 6.0 | 0.12, 0.10 | 1.6, 2.0 | 50 | 10 | – | speed, mass, deadband |
| DFR_CLUTCH_9G_300 | analog | 300 | 4.8, 6.0 | 0.12, 0.10 | 1.0, 1.2 | 50 | 10 | – | speed, mass, deadband |
| DFR_CLUTCH_2KG | analog | 180 | 4.8, 6.0 | 0.12, 0.10 | 1.6, 2.0 | 50 | 10 | – | speed, mass, deadband |
| DFR_CLUTCH_6KG | analog | 180 | 6.0 | 0.16 | 6.0 | 50 | 5 | – | speed, mass, deadband |

**SG90** is the scan servo default. 360° continuous-rotation variants are left out: their pulse sets speed, not angle.

Measured step response (8° roll step to 90%): SG90 80 ms, MG996R 100 ms, DS3218 90 ms, TD8135MG 165 ms (`tools/calibrate.py`).

---

## 9. Platform kinematics

`bop_env/legs.py`, `config/platform.toml`:
- servo axes on a 90 mm base circle, 40 mm above the base;
- 30 mm cranks, 80 mm rods;
- ball joints on a 120 mm circle under the plate;
- crank travel ±80°.

- **Zero = flat:** at servo angle 0 the crank is horizontal and the rod vertical, so `attach_radius = base_radius + crank`, and the plate is flat at neutral height. This is verified: `ik(0, 0, 0) ≈ 0`. Under load, deadband and backlash let it sag 0.14 mm.
- `ik(pf, plate, roll, pitch, heave) → (q[3], knee[3])`: closed form, per leg.
- `pose(xmat, xpos, pf, plate) → (roll, pitch, heave)` from the plate body.
- In MJCF the rods connect to the plate through equality constraints (`joint_solref`). The plate is a free body.

---

## 10. Scan and ToF sensor model

### Scan schedule (`scan.py`, `config/scan.toml`)
- Each corner sensor sweeps its 90° quadrant out and back as a triangle wave, with one 10 ms reading per 18° step (5 readings per sweep, 50 ms one way).
- Sensors are phase-staggered by a quarter sweep (`phase_stagger`) so diagonal sensors don't face each other.
- `mode = "step"` holds still during each reading instead.
- Scan servo angle 0 = beam on the corner bisector.

### Scan actuator (`scan.actuator`)
- `"servo"` (default): each housing sits on a hobby servo (`scan.servo_model` SG90 at `supply_v` 5 V, 50 Hz frames) driven by the same model as §8, carrying `rotor_mass_kg` (4 g). The triangle wave is sent as PWM targets. The beam yaw is the actual horn angle.
- `"ideal"`: the yaw follows the triangle wave exactly. This is equivalent to a stepper or galvo.

> Hobby servos cannot follow a 90°/50 ms sweep: it needs ~1800°/s against ~430–600°/s rated. Commanded ±45°, actual span: SG90 −33…−21°, MG945 −12…−5°, DS3218 @333 Hz −3…+4°. With `"servo"` the sensors are nearly fixed near the bisector. Use `"ideal"`, or slow the sweep (`sensor.timing_budget_ms`, `scan.step_deg`, `scan.sector_deg`), to study a working scanner.

The tracker bearing is set by `scan.yaw_source`: `"measured"` means the horn angle plus feedback noise; `"commanded"` means the schedule yaw.

### VL53L4CD (`tof.py`, `config/sensor.toml`)
Each reading integrates over the 10 ms budget in 2.5 ms sub-samples (`scan.substep_ms`). Each sub-sample works like this:
1. 300 rays are cast over the detection cone (FoV 22° at 100 mm narrowing to 18° at 1 m; VCSEL illumination 16° at 1/e²), weighted by the emission profile.
2. Each hit returns `albedo · cos / r²` into a range histogram (2 mm bins).
3. With `interference` on, the other sensors' light adds ambient, both diffuse (their spot seen by this receiver) and direct (housing in view).

At the end of the reading:
1. The histogram is smoothed with a 20 mm pulse kernel and the strongest peak picked (`peak_select = "nearest"` is the alternative). The range is the centroid around that peak.
2. Sigma is computed from the datasheet anchor: `σ = σ_ref · √(S_ref/S) · √(33 ms / budget) · √(1 + ambient/S)`.
3. The reading is valid if `σ ≤ 15 mm`. The range then gets Gaussian noise σ plus a per-unit offset drawn at each reset (`offset_sigma_m`, 2 mm).

S_ref (the white 88% wall at 200 mm) is computed at start-up with `tof.reference_signal`.

---

## 11. EKF tracker

`tracker.py`, `config/tracker.toml`. The state is `[x, y, vx, vy]` in the plate frame.
- **Predict** every physics step, with known acceleration `roll_factor · g` projected on the tilted plate (0.6 for a hollow ball) and white-acceleration noise `process_accel_sigma` (6 m/s²).
- **Update** per valid reading, with a range-bearing measurement from the sensor's corner:
  - The range sigma is the reported sigma, floored at 4 mm.
  - The bearing sigma comes from the cone half-width plus the ball's angular size.
- **Background rejection:** at construction, `Sim` records an empty-plate range map per sensor per scan slot (3 noise-free sweeps, ball parked away). A reading not at least 25 mm nearer than the background is rejected.
- **Gating:** χ² 9.21 (99%, 2 dof).
- **Initialisation:** from one reading. The fix is dropped after 0.3 s without updates (`tracked` → 0).
- The estimate is clamped inside the plate, which removes the mirror solution of two-range trilateration.

Status per reading (in `Sim.log` / `Sim.new`): `invalid`, `background`, `init`, `gated`, `accepted`.

---

## 12. Disturbances

`[disturb]` in `env.toml`, applied as a horizontal force on the ball in the plate frame:
- **Kicks:** Poisson at `kicks_per_s` (0.2/s), size up to `kick_force_n` (0.03 N) in a random direction, lasting `kick_ms` (20 ms).
- **Push:** Ornstein–Uhlenbeck force with σ `push_sigma_n` (0.3 mN) and time constant `push_tau_s` (0.3 s). It models air and plate unevenness.

Set them to 0 for a clean env. Raise them for robustness training.

---

## 13. Timing of one step

`env.step(a)` at `control_hz = 50`:
1. `send_legs(a · range)`: the target pulse is stored and latched at the next PWM frame.
2. 80 physics steps of 0.25 ms. Each one does the following, in order:
   - plate-servo torques;
   - scan-servo targets and torques;
   - disturbance force;
   - fall-off check;
   - a ToF sub-sample at mid-slice (every 2.5 ms); finishing a 10 ms reading triggers an EKF update;
   - EKF predict;
   - `mj_step`.
3. New readings are pushed into `obs["tof"]`, ages are increased, the reward is computed.

The loop stops early if the ball falls off. Latency in the observation: a reading is at most `age_s` old, and the EKF state is current to the end of the step.

---

## 14. Parallelism and hardware acceleration

- **Physics on CPU:** MuJoCo, ray casting and the EKF run in numpy, one process per env. A single env runs at ~45 steps/s (0.9× real time). Most of the time goes to the ToF ray casts (~50%), then `mj_step` (~14%) and the servo model (~8%).
- **Many headless copies:**
  - `bop_env.make_vec(n)` gives a Gymnasium `AsyncVectorEnv` with the `forkserver` start method; `asynchronous=False` gives a `SyncVectorEnv`.
  - Under SB3, `control/train.py` uses `SubprocVecEnv` (forkserver) plus `VecMonitor` and `VecNormalize`.
  - 14 workers give ~335 steps/s.
  - Scripts that start workers need an `if __name__ == "__main__":` guard and must be run from a file, not stdin.
- **Neural network on GPU:** the policy trains with torch CUDA (`device = "cuda"` in `control.toml`).
- **Not done:** GPU physics (MJX / MuJoCo Warp). It would need the ToF histogram model and the EKF rewritten in JAX or Warp.
- **Rendering:** headless unless `render_mode="human"`, which opens the passive viewer.

Training and evaluation:

```bash
.venv/bin/python control/train.py --algo ppo --n-envs 14 --steps 2000000   # ppo|sac|td3|a2c|ddpg
.venv/bin/python control/evaluate.py --controller pid --episodes 10
.venv/bin/python control/evaluate.py --controller model --path runs/<run>/final.zip
```

Baselines (4 episodes, defaults):

| Controller | Result |
|---|---|
| PID | never falls; 26–29 mm RMS from centre; estimate error ~6 mm |
| Zero action | ball falls off in ~2 s |

---

## 15. Using Sim directly

`Sim(cfg, interference=None, keep_log=False)` is the physics loop without the Gymnasium wrapper. `tools/` uses it.

```python
from bop_env import Sim, load_all
sim = Sim(load_all({"scan.actuator": "ideal"}))
sim.reset(ball_xy=(0.05, 0), tilt=(0, 0), seed=1)
sim.command_pose((0.0, 0.03))          # roll, pitch (rad) through the IK
readings = sim.advance(0.5)            # list of reading dicts
p, v = sim.ball()                      # true ball, plate frame
sim.trk.x, sim.trk.ok                  # EKF state, fix flag
```

| Method / attribute | Meaning |
|---|---|
| `reset(ball_xy, ball_v, ball_z, tilt, heave, forces, horizon, seed)` | New episode. |
| `send_legs(q)` / `command_pose(tilt, heave)` | Plate servo command, by angle or by pose via IK. |
| `step()` / `advance(seconds)` | One physics step / many; `advance` returns the finished readings. |
| `ball()`, `plate_pose()`, `leg_angles()`, `scan_angles()`, `leg_feedback()` | State. |
| `sensor_poses()`, `in_cone(rd)` | Beam frames; was the true ball inside the cone of a reading. |
| `escaped`, `legs.saturated`, `n_saturated` | Fall-off flag, servo saturation. |
| `log` (if `keep_log`), `new`, `last` | Readings, with `true_range`, `true_bearing`, `beam_yaw`, `status`. |
| `m`, `d`, `ids` | MuJoCo model, data and the index map. |

---

## 16. Known limits

- **Scan servos are too slow for the default sweep (§10).** Results with `scan.actuator = "servo"` describe near-fixed sensors.
- The plate pose observation is noise-free (no IMU model).
- The ball is a perfect sphere and the plate is flat and uniform. Spin and slip come from MuJoCo's soft contact.
- No supply sag: `supply_v` is constant even when all servos draw stall current.
- Servo internals (P band, backlash, time constant, friction) are assumed, not measured. They are named in `servos.toml`.
- The default reward uses the true ball state. Set `reward_source = "estimate"` for a reward computable on hardware.
- A model rebuild (randomisation, `options["overrides"]`) costs about 0.35 s, mostly for the background map.
