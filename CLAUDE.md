# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All Python runs from the project venv (`.venv`, made with uv; deps in `requirements.txt`; `bop_env` installed editable from `pyproject.toml`).

```bash
uv venv .venv -p 3.13 && uv pip install --python .venv/bin/python -r requirements.txt   # includes -e .
# W&B key: WANDB_API_KEY in git-ignored .env (template .env.example); train.py logs to W&B ([wandb] in control.toml, --no-wandb / --wandb-mode offline)
.venv/bin/python tools/calibrate.py                 # sanity checks: ToF vs datasheet, rolling physics, servo step times
.venv/bin/python tools/run_scenarios.py             # scripted scenarios, interference on + off -> sim/out/
.venv/bin/python tools/run_scenarios.py --scenario circle --interference on --view
.venv/bin/python tools/run_scenarios.py --set scan.actuator=ideal --out sim/out/ideal_scan
.venv/bin/python tools/ablate.py                    # variants in config/ablation.toml x scenarios x seeds, parallel
.venv/bin/python control/evaluate.py --controller pid --episodes 10     # non-learned baseline on the env
.venv/bin/python control/train.py --algo ppo --n-envs 14 --steps 2000000  # ppo|sac|td3|a2c|ddpg, NN on CUDA
.venv/bin/python control/evaluate.py --controller model --path runs/<run>/final.zip
.venv/bin/python control/train.py --resume runs/<run>/final.zip --steps 500000   # or best.zip / ckpt/<algo>_<n>_steps.zip (last 5 kept); Ctrl-C saves final.zip
```

See `docs/ENV.md` for the full env reference. `--set section.key=value` (JSON value) overrides any config value in tools/ and control/ scripts. No test suite; `gymnasium.utils.env_checker.check_env` passes on the env.

## Code style (user rules)

Max 500 lines per file, no heavy indirection, simple lines, short comments. Every parameter lives in `config/*.toml`, never hardcoded in scripts.

## Layout

Environment code and controller/learning code are kept apart (user rule):
- `bop_env/`: the simulation and the Gymnasium env (`BallOnPlateToF-v0`). Nothing here knows about learning.
- `tools/`: env studies with scripted tilt (scenarios, ablation, calibration, plots).
- `control/`: controllers acting through the env API. `pid.py` is the non-learned cascade PID (uses `bop_env.legs.ik`), `train.py` is SB3 training (`--net tof`: `tof_net.py`, 3-layer LayerNorm+dropout net on raw ToF only), `evaluate.py` scores any controller. Settings in `config/control.toml` (not read by the env).

## Simulation architecture (bop_env/)

MuJoCo sim of 1–4 corner VL53L4CD sensors (fitted corners in `scan.corners`) oscillating over 90° in 18° steps, phase-staggered, tracking a 40 mm ball on a 30 cm plate on a 3-RRS servo platform. The ideal gimbal mode was removed (2026-11-04).

- `config.py`: `load_all(overrides)` loads every `config/*.toml`. Overrides take dotted keys at any depth (`"servos.MG996R.deadband_us"`) or nested dicts. s_ref is computed on the fly (`tof.reference_signal`); there is no calibration file any more.
- `servo.py` + `config/servos.toml`: every positional servo on the roboticsbd listing (SG90, SG92R, MG995, MG996R/_270/_FB, MG945, DS3218, TD-8120MG/_270, TD-8135MG/_270, LDX-218/227, MAG25, DFRobot clutch servos). Listed values: travel, speed and stall torque vs voltage (interpolated at `supply_v`), deadband, mass, frame rate, feedback wire. Values not on the listing are named in each model's `assumed` list; generic assumed values are in `[default]` (P band, backlash, mechanical time constant, gear friction, feedback noise).
  - Host side: angle → pulse (centre 1500 µs = angle 0), quantised to `pwm_resolution_us`, latched once per PWM frame (`servo_update_hz`, capped at the model's `max_frame_hz`). An optional host speed limit (`trajectory_speed_rps`) is the only speed control.
  - Servo side: P loop on the horn angle with deadband → DC motor, torque = T_stall·(duty − ω/ω_noload). Speed tops out at the rated no-load speed and torque at stall.
  - In the MJCF each servo is a motor hinge (reflected armature, gear friction) in series with a free backlash hinge with end stops. Horn angle = motor + backlash.
- `legs.py`: 3-RRS geometry, closed-form `ik()`, MJCF for the legs. **Servo angle 0 = crank horizontal = plate flat** at neutral height (verified: `ik(0,0,0) = 0`).
- `model.py`: builds the MJCF. Plate is a free body on the legs. The rim is optional (`rim_height_m = 0` gives an open edge). The ball is free (hollow-shell inertia). Each sensor housing is one body. With `scan.actuator = "servo"` it sits on a scan servo (z hinge, angle 0 = corner bisector); with `"ideal"` the yaw follows the triangle wave exactly.
- `sim.py`: `Sim` is the one physics loop shared by the env and tools. It handles `reset()`, `step()` and `advance(s)`, with host commands `send_legs(q)` and `command_pose(tilt, heave)`. Each step runs, in order: servo torques, scan servos, random ball forces, ToF sub-samples, EKF predict, then `mj_step`. It builds the empty-plate background map on construction. `escaped` becomes true when the ball centre passes the plate edge or drops below the plate top.
- `tof.py`, `scan.py`, `tracker.py`, `disturb.py`: as before. `tof.py` covers the ray-cast VL53L4CD (histogram, strongest peak, sigma from signal, interference as ambient). `scan.py` is the triangle-wave schedule, `tracker.py` the EKF with a tilt-aware process and a clamp inside the plate, and `disturb.py` the tilt profiles plus kicks and OU push. The tracker bearing is the commanded yaw or the measured horn angle plus feedback noise (`scan.yaw_source`).
- `env.py`: `BallOnPlateEnv(overrides, render_mode)` and `make_vec(n)` (gymnasium `AsyncVectorEnv`, forkserver).
  - Action: 3 servo angles / `platform.range_deg`, in [-1, 1].
  - Observation: a Dict in SI units: `ball_est` [x, y, vx, vy, tracked], `tof` [sensor, history, (valid, range, sigma, signal/S_ref, ambient/S_ref, yaw, age)], `servo_cmd`, `target`, plus optional `servo_fb` (feedback servo models only) and `plate` pose.
  - `env.overrides` removes the rim, so the ball falls off the edge. It then returns `-escape_penalty` and the episode terminates.
  - Each step is one 20 ms control period by default (`control_hz` 50). Reward uses the true ball position by default.
  - `[randomize]` ranges are redrawn every reset, which rebuilds the model.
- `viewer.py`: passive viewer, used by `tools/run_scenarios.py --view` and `render_mode="human"`.

Hardware acceleration: physics, ray-cast ToF and the EKF are CPU (MuJoCo + numpy), one process per env. The policy network trains on the GPU (torch CUDA, RTX 3070). Throughput is ~45 env steps/s per process (0.9× real time) and ~335/s with 14 workers. A GPU physics port (MJX / MuJoCo Warp) would need the ToF histogram model and EKF rewritten in JAX/Warp and has not been done.

Servo refactor results (2026-11-04):
- Hobby scan servos cannot follow the 90°/50 ms sweep, which needs ~1800°/s against ~430–600°/s rated. Commanded ±45°, actual span: SG90 −33…−21°, MG945 −12…−5°, DS3218 @333 Hz −3…+4°. The sensors end up nearly fixed near the bisector.
- Scripted circle scenario: ideal scan 6.9 mm RMSE, SG90 scan 27 mm.
- Under PID balancing near the centre, the estimate error is 6 mm with servo scan and 7 mm with ideal scan. The PID holds the ball in 4/4 episodes (26–29 mm RMS from centre). With zero action the ball falls off in ~2 s.
- Physics check: 2° tilt ball acceleration within 1% of theory. 8° roll step reaches 90% in: SG90 80 ms, MG996R 100 ms, DS3218 90 ms, TD-8135MG 165 ms.
- Older results below were run with `sim/` scripts, since replaced by `bop_env/` + `tools/`.

PWM/servo-model ablation (2026-11-01, 234 runs). Mean plate tilt error RMS: MG996R @50 Hz 0.47°, DS3218/TD-8120MG @50 Hz 0.46°, 333 Hz 0.33°, continuous 0.31°, ideal gimbal 0.27°.
- The 50 Hz frame hold costs ~0.15° RMS. Faster servos barely help, because torque never limits.
- Ball RMSE (12.4–13.9 mm mean) is dominated by run-to-run trajectory variance in the random scenarios, not by servo choice.

Servo platform (2026-11-01, 429-run ablation, random tilt now zero-mean with a 1 s ramp-in):
- Plate follows commanded tilt within 0.0–0.8° RMS. Max 3.5° in random_tilt_fast, where the servo speed limit is active 5% of the time. Heave error < 0.5 mm.
- 4 sensors on servo3: mean RMSE 12.6 mm, worst 34.7 (random_tilt_fast). Ideal gimbal: 13.9 / 44.6. The gimbal is worse because it follows fast tilt without a speed limit, so the ball moves faster.
- Sensor ranking unchanged: 4 (12.6) < 3 (15.2) < 2 diagonal (16.8) ≈ 2 adjacent (18.0) < 1 (28.5).

Ablation (2026-11-01, 330 runs, 3 seeds, 10 s each, bumped disturbances: tilts up to 12°, kicks up to 0.07 N, 50 cm drop). Mean / worst RMSE over 11 scenarios:
- 4 sensors: 13.5 / 30.7 mm. 3 sensors: 15.6 / 39.1. 2 adjacent: 17.7 / 34.9. 2 diagonal: 17.6 / 36.2. 1 sensor: 29.1 / 62.5.
- 4 sensors, single factors: no stagger 14.5. Step-and-hold 13.1. 10° steps 14.7. Fixed 18° FoV 13.0. No interference 13.6. All within ~1 mm except no stagger and 10° steps.
- Calm scenarios: 5–12 mm. Strong kicks and random tilt (ball up to 0.8 m/s): 17–31 mm, p95 up to 68 mm.
- 50 cm drop: 7.4 mm once the ball is back on the plate. Ball contact `ball_solref` gives restitution 0.85. Drop onto a tilted plate makes the ball bounce over the rim (real physics), so the drop scenario releases over a flat plate and starts tilting after `motion_delay_s`.
- Older single-config results below were run before the disturbance bump.

Key results (2026-11-01, 10 s runs, tracker process_accel_sigma 6 from sweep):
- Calm runs: RMSE 4–8 mm. With kicks, steep tilts up to 8° and ball speeds up to 0.5 m/s: RMSE 7–10 mm, p95 ≤ 19 mm, max ~31 mm.
- Worst case: ball resting at the rim near the line joining two sensors (hold_x2), RMSE 13.5 mm, because trilateration geometry is poor there.
- 45–135 accepted readings/s. The ball is detected in 53–85% of in-cone readings, since 10 ms budgets lose small far targets.
- Interference is negligible with the phase stagger. With `--no-stagger`, the diagonal sensors blind each other (14% of readings invalid).
- Plate is driven by position servos (kinematic qpos override blew up contacts at fast tilts). Rim is 25 mm so the ball can't roll over it. The run stops and sets `escaped_at_s` if the ball leaves the plate.
- `disturb.py` handles tilt profiles (hold/sine/circle/random) and random ball forces (kicks plus a slow push), configured per scenario.

## Project goal

Research project: **closed-loop 2-D ball-on-plate balancing using time-of-flight (ToF) ranging** for ball localisation. Source of truth for goals and literature: systematic review doc "ToF Sensing for Ball-on-Plate Balancing: Systematic Review" (https://claude.ai/artifact/4ZqQNAEhsqRX3zVobEzF6g, as of 2026-10-31). Read it with the Claude Docs tools, not web fetch.

### Key findings driving design

- No closed-loop 2-D ball-on-plate system using ToF was found. ToF (VL53L0X) only appears on 1-D ball-and-beam builds with no reported accuracy/latency.
- Closest prior work: Bayar (2021), Hokuyo rotating laser scanner on Quanser 2-DOF balancer — detection only, 10 Hz, scanner was the bottleneck.
- **Rotating single-ToF design is rejected**: scanning divides sample rate across angles, `f_update = f_sample / N_angles`, so `f_sample ≥ f_control × N`. VL53L4CD at 10° steps ≈ 2.8 updates/s. Report it as the motivating negative result.
- Rotating feasibility detail is in the doc's "Rotating ToF feasibility" tab (https://claude.ai/artifact/4ZqQNAEhsqRX3zVobEzF6g#e3e89303-fc24). Target bar there: ≥50 Hz. Result: no single rotating sensor passes, and neither does any continuously spinning layout, because ¾ of each turn faces away from the tray. A sensor's reading rate (VL53L4CD 100 Hz) is not its position rate: update rate = useful readings/s ÷ readings per sweep. The most promising rotating layout is 4× VL53L4CD, one per corner, each on a servo that keeps its beam on the ball, with positions from trilateration of the corner ranges (100 Hz). Oscillating those servos over 90° instead, phase-staggered, gives ≈44 Hz at 10° steps or ≈80 Hz at 18° steps. 4× STL-27L phase-staggered gives ≈52 Hz but is bulky and unproven. A single STL-27L (21.6 kHz samples, 0.17° steps, ±15 mm σ5 mm) is motor-capped at 13 Hz, so it serves as a reference sensor. Key constraints: the 2° lidar pitch tolerance gives ±15 mm of beam height over the tray, and smear is negligible (~1 ms per ball crossing) while staleness (77 ms) dominates.
- Simulation engine is MuJoCo (user choice, 2026-11-01). See Simulation architecture below. Ask the user before changing scenarios or physics assumptions.
- Viable routes are **fixed sensors**:
  - Two single-zone VL53L4CD at adjacent corners + trilateration (up to 100 Hz). Error degrades near the line joining the sensors.
  - One multizone VL53L5CX on an edge: zone index → bearing, reading → range. 4×4 @ 60 Hz vs 8×8 @ 15 Hz. Known offset ~18–32 mm; small/partial-zone targets (e.g. 40 mm ball) untested.

### Physical constraints

- Beam must cross ball near its equator, a few cm above plate → mount sensors on the plate so they tilt with it.
- Motion smear: at 0.5 m/s ball moves 50 mm in a 0.1 s scan.

### Sensor reference

| Sensor | Rate | FoV | Short-range accuracy |
| --- | --- | --- | --- |
| VL53L4CD | ≤100 Hz | 18° (spot 32 mm @100 mm, 48 mm @150 mm) | ±7–8 mm < 200 mm |
| VL53L0X | ~45 Hz @ 20 ms budget | ~25° | not reported |
| VL53L5CX | 60 Hz (4×4), 15 Hz (8×8) | 65° diag | offset 18–32 mm |

### Planned contributions (ascending risk)

1. Characterisation: VL53L4CD / VL53L5CX error, dropout, latency on 30–60 mm balls of varied finish across a 30 cm plate, incl. partial-zone case.
2. Fixed two-corner trilateration, closed loop up to 100 Hz, with error map.
3. Single-edge multizone localisation (VL53L5CX), compare 4×4 vs 8×8.
4. Benchmark: same plate + controller with ToF vs resistive touchscreen vs camera — position noise, latency, velocity-estimate quality, closed-loop settling. Optionally sim-to-real transfer per sensor (links to RL work / H3 in omni-base review).

Evaluation metrics throughout: position accuracy, update rate, latency, noise/dropout, control performance.

### Open next step

Confirm the gap with a Scopus / IEEE Xplore database query (only a web sweep was done). Absence claims mean "not found in this sweep".
