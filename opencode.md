# Codebase analysis

## How to read this document

> Implementation note (2026-11-16): the repository now includes an optional batched MuJoCo Warp backend in `bop_env/gpu/`, and `control/train.py` defaults to `--physics warp`. The CPU simulator remains the reference. The findings and recommendations below were written before that backend existed; the GPU-related "not done" statements describe the state at the time of the review.

This document records a static review of the ball-on-plate research codebase. It explains what the project does, how its major components fit together, which parts are particularly strong, and which issues should be addressed before the results are used to support scientific conclusions. The findings are ordered by their likely effect on correctness and reproducibility rather than by how difficult they are to fix.

This was a read-only code review. No simulations, training runs, or numerical experiments were performed during the analysis, and the historical results quoted by the project documentation were not independently reproduced. The distinction matters: a finding described below as a bug follows directly from the source code, while a modeling limitation describes an assumption that may be reasonable but must be disclosed when interpreting results.

## Overall assessment

This repository is a well-structured, advanced research prototype. Its strongest design decision is that scripted experiments, PID control, and reinforcement learning all use the same `Sim` physics loop. This avoids a common research-code problem in which open-loop sensor studies and closed-loop control experiments gradually acquire different physical assumptions.

The code is generally small, readable, and sensibly separated into simulation, tooling, configuration, and control. The servo and time-of-flight models contain considerably more physical detail than a typical ball-on-plate demonstration. The documentation is also unusually candid about assumptions, incomplete work, and the slow scan-servo result.

The main risks are not architectural. They are several concrete correctness bugs, fragile off-policy training resume behavior, weak configuration validation, and insufficient experiment provenance and automated testing. These should be addressed before investing in long training runs or making strong comparative claims from ablation studies.

## Architecture and data flow

The codebase has four principal layers:

| Area | Responsibility |
| --- | --- |
| `bop_env/` | Builds and advances the MuJoCo simulation, models the platform and sensors, estimates the ball state, and exposes the Gymnasium environment. It deliberately contains no learning code. |
| `tools/` | Runs scripted scenarios, paired interference studies, parallel ablations, calibration checks, and plotting. |
| `control/` | Implements the PID controller, raw-ToF neural network, Stable-Baselines3 training, checkpoint management, and evaluation. |
| `config/` | Stores the physical, sensing, scanning, environment, scenario, and training settings used by these components. |

This table shows a clean separation between the system being studied and the algorithms used to control it. In particular, the simulation can be reused with another controller or learning library without moving physics code into the controller.

At runtime, `bop_env/config.py` loads the TOML sections and applies overrides. `bop_env/model.py`, `legs.py`, and `servo.py` construct the plate, ball, 3-RRS legs, and actuator dynamics. `bop_env/sim.py` then coordinates scan scheduling in `scan.py`, optical measurements in `tof.py`, disturbances in `disturb.py`, and state estimation in `tracker.py`.

Each `Sim.step()` performs the following sequence: it calculates plate-servo torques, commands and updates the scan servos, applies disturbances, checks whether the ball has escaped, takes any scheduled ToF subsample, advances the EKF prediction, and finally calls `mj_step`. This ordering is visible in `bop_env/sim.py:102-122`.

`bop_env/env.py` wraps that simulation in a bounded continuous-action Gymnasium task. The environment advances normally 20 milliseconds per control action and returns the EKF estimate, raw ToF histories, the previous command, and the target, with optional servo feedback and plate pose. `control/train.py` connects this interface to Stable-Baselines3, while `control/evaluate.py` provides a common evaluation route for PID, zero-action, and learned policies.

## Main strengths

### One authoritative simulation

Both `tools/` and `bop_env/env.py` use `Sim`. This is a substantial methodological strength because the sensing behavior measured in a scenario is, by construction, the same sensing behavior presented to a controller.

### Detailed actuator modeling

The servo implementation includes pulse quantization, PWM frame latching, deadband, backlash, voltage-dependent specifications, internal proportional control, and torque-speed behavior. Several actuator properties are explicitly marked as assumed where manufacturer data is unavailable, which is preferable to presenting every number as measured fact.

### Physics-based ToF sensing

The sensor model uses ray geometry, emitter and receiver profiles, return strength, histogram peak selection, background rejection, and interference. This supports meaningful studies of geometry, dropout, and scan timing instead of reducing every sensing problem to additive Gaussian noise.

### Raw and filtered observations

The environment exposes both raw ToF histories and the EKF estimate. This makes it possible to compare a conventional estimator-plus-controller pipeline with direct learning from sensor data while retaining the same task and physics.

### Readable implementation and documentation

The environment modules are compact and have clear responsibilities. The documentation generally matches the implementation, particularly in its description of the observation fields, reward terms, configuration precedence, and inability of hobby scan servos to follow the commanded sweep.

## Prioritized findings

The following table summarizes the issues with the highest expected impact. Subsequent sections explain their mechanism and consequences in more detail.

| Priority | Finding | Main consequence |
| --- | --- | --- |
| High | Tracking loss does not expire during invalid or absent readings. | A stale estimate can remain marked as tracked indefinitely, affecting control and reported estimator metrics. |
| High | The reported plate-frame velocity is not relative to the moving plate. | The default reward can penalize velocity caused by platform motion rather than ball motion relative to the plate. |
| High | Off-policy resume can train immediately on an almost empty replay buffer. | Resumed SAC, TD3, or DDPG training can severely disrupt the learned networks. |
| High | Changing the worker count on resume is unsafe for replay and action-noise shapes. | A resume can fail or handle experience inconsistently. |
| Medium | Configuration keys, enums, and timing relationships are not validated. | Typographical errors can silently change or invalidate an experiment. |
| Medium | Evaluation continues without required observation-normalization statistics. | A model can be evaluated on the wrong input distribution without a warning. |
| Medium | `best.zip` is selected using exploratory training return. | It is not necessarily the best deterministic controller on held-out conditions. |

### 1. Tracking loss does not expire correctly

`Tracker.update()` returns immediately when a reading is invalid or classified as background. The `lost_after_s` check occurs only after those early returns (`bop_env/tracker.py:52-65`). If no readings complete, `update()` is not called at all. `Tracker.predict()` also does not check the time since the last accepted update (`bop_env/tracker.py:34-42`).

Consequently, a complete sensing failure does not reliably clear the `ok` flag. The environment can continue reporting `tracked = true` even though the estimate has not been corrected for much longer than the configured timeout. This affects the observation given to controllers, the conditional estimator RMSE in evaluation, and scenario tracking statistics.

The timeout should be evaluated from simulation time during prediction, or through an equivalent path that runs regardless of whether a new reading arrives. The documented behavior is that a fix is dropped after 0.3 seconds without updates; the current implementation does not guarantee that invariant.

### 2. True plate-frame velocity is incomplete

`ball_plate_frame()` computes position as a plate-relative displacement, but computes velocity only by rotating the ball's world linear velocity into plate coordinates (`bop_env/model.py:160-165`).

For a rotating and translating plate, the derivative of plate-relative position also requires subtraction of the plate origin velocity and the rotational transport term. In symbols, if $R$ maps plate coordinates to world coordinates, $x_b$ and $x_p$ are the ball and plate origin positions, and $\omega_p$ is the plate's world angular velocity, the relative velocity is

$$
\dot p = R^T\left(v_b-v_p-\omega_p\times(x_b-x_p)\right).
$$

The present function returns only $R^T v_b$. Therefore, a ball stationary relative to a moving plate can be reported as having nonzero plate-frame velocity.

The default truth-based reward uses this quantity in its velocity penalty (`bop_env/env.py:135-144`), as does `info["ball_v"]` (`bop_env/env.py:146-151`). Position errors and position RMSE are unaffected. The default PID derivative uses the EKF estimate rather than this true velocity, so this finding does not imply the same defect in the PID derivative calculation.

### 3. Off-policy resume is unsafe without replay data

On resume, `control/train.py` loads the model and its old timestep, then loads a replay buffer only if the expected sidecar exists (`control/train.py:127-133`). Checkpoint replay saving is disabled by default, and the normal best-model path does not create a replay sidecar.

Stable-Baselines3 therefore sees a model whose global timestep is already beyond `learning_starts`, while the replay buffer may contain only newly collected transitions. Gradient updates can begin before a meaningful amount of new experience has accumulated. This is more dangerous than merely requiring a short refill period: repeated updates on a tiny buffer can rapidly damage critics and the actor.

Before learning resumes, the training script should either restore a valid replay buffer or enforce a fresh warmup based on the actual replay-buffer size. This applies to SAC, TD3, and DDPG.

### 4. Worker-count changes are incompatible with some saved state

The resume CLI permits a new `--n-envs`, but a saved off-policy replay buffer retains the vector dimension from the original run. Loading it into an environment with a different worker count can lead to broadcasting or shape failures. TD3 and DDPG can additionally retain vectorized action noise with the original worker dimension.

The simplest safe behavior is to require the original worker count whenever vector-shaped replay or action-noise state is restored. A more complicated migration is unnecessary unless there is a demonstrated need for it.

### 5. Configuration errors can silently change experiments

`config.apply()` accepts unknown keys and creates missing sections (`bop_env/config.py:29-40`). This makes a misspelled section or parameter indistinguishable from a valid override. For example, an incorrectly spelled platform section can be stored without ever affecting the platform.

There are also no checks for valid enumerated values or compatible timing relationships. Of particular concern, only the exact string `servo` constructs a servo-driven scan; other values effectively take the ideal-scan path (`bop_env/sim.py:23-25`). A typo can therefore alter the physical mechanism being studied instead of producing an error.

Validation should cover known keys, types, allowed strings, unique sensor corners, positive integer timing divisors, and compatible control and physics periods. Unknown-key rejection is especially important because experimental overrides are central to the workflow.

### 6. Evaluation silently accepts missing normalization

`load_model()` returns no normalization object when the corresponding file is missing (`control/evaluate.py:47-55`). `ModelPolicy` then sends unnormalized observations to the model (`control/evaluate.py:79-87`). For a network trained with normalized observations, this is not a safe fallback; it evaluates a different input distribution and may produce misleading behavior.

Evaluation should fail loudly when an expected normalization sidecar is missing. It should also verify observation keys and shapes against the checkpoint.

### 7. Best-model selection is not held-out evaluation

`best.zip` is selected from rolling training episode returns. These episodes use the training distribution and, for algorithms such as SAC, stochastic exploration. The rolling buffer can also combine episodes produced by different policy parameters.

The artifact is therefore accurately described as the **best observed training-return checkpoint**, not necessarily the best controller. A separate evaluation environment with frozen observation normalization, deterministic actions, and fixed held-out seeds would provide a defensible selection criterion. Survival and true-position error should accompany return in that criterion.

## Other correctness and robustness concerns

The following issues are less central than those above but still worth addressing:

- The EKF fuses an integrated ToF reading at the window boundary rather than accounting for its effective measurement time. This can introduce motion-dependent registration error.
- Tracker position clamping changes position and velocity without a corresponding covariance projection (`bop_env/tracker.py:26-32`).
- Existing observation ages are advanced by the full control period even when an episode terminates partway through that period (`bop_env/env.py:112-122`).
- Scenario and ablation filters can produce empty result sets and then fail while accessing the first row instead of explaining an unknown scenario or variant name.
- Very short runs, immediate escapes, and zero-reading runs have unchecked empty-array and division edge cases in scenario metrics.
- Ablation outputs are written only after the whole process pool finishes, so an interruption can discard a large study. Existing output directories can also be overwritten without a protective option.
- Several scientific and plotting constants remain hardcoded, including calibration cases and a 20 mm plotting radius, despite the general convention that parameters belong in TOML.
- A non-editable package installation does not reliably include the external `config/` directory. The documented editable repository workflow works, but standalone packaging is incomplete.
- There is no automated test suite or continuous integration, and there is no lockfile capturing transitive dependencies.

## Modeling limitations to keep explicit

These items are not automatically implementation defects. They are simplifications whose effect depends on the research question.

The EKF uses a tilt-aware constant-acceleration model but omits plate angular acceleration, Coriolis and centripetal terms, impacts, slip, and airborne phases. Its configured rolling factor is also independent of the ball-inertia option in the physical model.

The ToF interference model adds ambient and noise but does not reproduce spurious correlation peaks or device-specific firmware behavior. Broad-cone bearing and off-axis sphere-range effects are approximated. The background map is static and indexed by nominal scan slot even when actual servo motion lags the schedule.

The default SG90 scan servos cannot follow the planned sweep, so the servo-scanned configuration behaves much closer to nearly fixed beams than to the nominal trajectory. This limitation is already documented and should remain prominent in every comparison with ideal scanning.

Measured scan yaw is available to the simulated tracker even for an SG90 configuration whose model declares no host feedback interface. A hardware implementation would need an explicit angle measurement mechanism or would have to rely on commanded yaw.

The default simulated reward uses privileged true ball state. This is valid for simulation training and is not a direct observation leak into the policy, but it is not an online hardware reward. Plate-pose observations are also noise-free.

## Experiment maturity and reproducibility

The repository is best characterized as a strong experimental scaffold rather than a finished, publication-grade experiment platform. The physics and sensing work is more developed than the experiment-management infrastructure.

The current project status documentation correctly notes that no full neural-network training comparison has been completed, older tracking experiments predate the newest servo model, no hardware has been built, and the formal Scopus and IEEE Xplore confirmation remains open. The existing `sac_tof_1m` artifacts should not be interpreted as a completed one-million-step result.

Generated studies should carry a manifest containing the complete resolved configuration, original command-line arguments, seeds, source revision, dependency versions, result schema version, and resume history. Saving only override deltas means later edits to default TOML files can silently change the task associated with an old checkpoint.

Evaluation should use several independent training seeds and common held-out evaluation seeds. Reports should include survival or fall-off rate, position RMSE, final target error, episode duration, tracking coverage, loss duration, and estimator error. Conditional estimator RMSE should always be accompanied by coverage-aware metrics so that a tracker is not rewarded for succeeding only during easy intervals.

## Recommended implementation order

1. **Fix correctness first.** Repair tracking timeout behavior and plate-relative velocity. Add safe replay-less warmup, reject incompatible resume worker counts, and require normalization sidecars during evaluation.
2. **Add configuration validation.** Reject unknown keys and invalid enum values, and validate timing and geometry constraints before constructing simulations.
3. **Add focused regression tests.** Cover tracking loss, relative velocity, checkpoint sidecars, resume behavior, configuration failures, early escapes, empty metrics, and short deterministic scenarios.
4. **Make experiments self-contained.** Save resolved configurations and source provenance, make long ablations incremental and resumable, and protect existing outputs from accidental overwriting.
5. **Improve model selection and evaluation.** Use deterministic held-out evaluation with fixed seeds, and report variability and failure-aware sensing metrics.
6. **Improve the raw-ToF policy input.** Include the previous servo command and target so the network has actuator context and can represent varying-target tasks.
7. **Run long comparisons only after the previous steps.** Compare PID, EKF-based learning, raw-ToF learning, and all-observation learning under the same evaluation conditions.
8. **Proceed to the remaining research goals.** Regenerate older studies, complete the formal database search, characterize real sensors, and evaluate sim-to-real transfer when hardware becomes available.

## Bottom line

The core architecture is sound and should be preserved. The shared simulation, detailed actuator model, physics-based sensing, and clean controller separation are strong foundations. The immediate priority is to make tracking, velocity, resume, and normalization behavior correct, then add validation, provenance, regression tests, and held-out evaluation. Those changes will contribute more to the reliability of the research than additional model complexity or longer training runs completed before the fixes.

## Glossary

| Term | Meaning in this project |
| --- | --- |
| **3-RRS platform** | A plate supported by three legs, each using a rotary actuator, a rotary joint, and a spherical joint to produce plate tilt and heave. |
| **Ablation** | An experiment that changes one design factor, such as sensor count or scan method, while attempting to hold other conditions constant. |
| **EKF** | An extended Kalman filter. It combines a motion prediction with noisy range and bearing measurements to estimate ball position and velocity. |
| **Held-out evaluation** | Testing on predetermined conditions that were not used to select or tune a model. |
| **Off-policy learning** | Reinforcement learning that can train from transitions collected by earlier versions of a policy. SAC, TD3, and DDPG use a replay buffer for this purpose. |
| **Replay buffer** | Stored environment transitions sampled during off-policy training. |
| **RMSE** | Root-mean-square error, a summary of the typical magnitude of position or estimation error. |
| **ToF** | Time of flight. Here it refers to optical ranging using VL53L4CD-like sensors. |
| **VecNormalize** | Stable-Baselines3 state that normalizes observations and, optionally, rewards using running statistics. A trained model needs the matching statistics during evaluation. |
