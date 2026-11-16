# Ball-on-Plate Balancing with Time-of-Flight Sensors

**Project status and technical documentation**

*Last updated: 13 November 2026. Code repository: `git@github.com:azfarus/tof-ball-on-plate.git`*

---

## How to read this document

This document explains the whole project from the ground up, for a reader who has not seen it before. It is written in three layers:

- **Sections 1 to 4** explain the idea: what problem we are solving, why it is interesting, and what the system looks like. No mathematics is needed.
- **Sections 5 to 9** explain how the simulation and the learning setup work, what results we have so far, and how to run everything.
- **Section 10 onwards** covers where things are in the code, what is finished and what is not, and a glossary of terms.
- **Appendix A** gives the full mathematics behind every part of the simulation: the physics, the sensor model, how the ball's position is estimated, and how the sweep rates were derived. Each formula is introduced by a sentence that says what it means, so you can follow the argument even if you skip the algebra.

---

## 1. The project in a nutshell

A **ball-on-plate** system is a classic control-engineering demonstration. A ball sits on a flat plate, and motors tilt the plate to keep the ball at a chosen spot, usually the centre. It works like balancing a marble on a tray by tilting your hands. If the tray tilts a little to the left, the ball starts rolling left, so you tilt it back to the right to stop it.

To do this automatically, the controller must constantly know **where the ball is**. Existing ball-on-plate systems almost always find the ball with either a **camera** looking down from above, or a **resistive touchscreen** used as the plate itself, which feels where the ball presses.

This project asks a different question: **can we locate the ball with small, cheap time-of-flight (ToF) distance sensors mounted at the edges of the plate?** A ToF sensor fires a short pulse of invisible infrared light and measures how long the light takes to bounce back. That time tells it how far away the nearest object is. These sensors are tiny (a few millimetres), cost a few dollars, and need no camera, no image processing and no special plate surface.

Our search of the research literature found **no existing ball-on-plate system that uses ToF sensors in a closed loop**. Before building any hardware, we therefore built a detailed **computer simulation** to find out whether the idea can work, and which sensor arrangement works best. The simulation models the physics of the plate and ball, the behaviour of real hobby servo motors, and the optics of the ToF sensors themselves.

On top of the simulation we built a **reinforcement-learning environment**. In it, a neural network can learn by trial and error to balance the ball, using nothing but the raw sensor readings.

---

## 2. Background: why this is an interesting problem

### 2.1 What a balancing controller needs to know

To keep a ball still on a plate, the controller has to answer two questions many times per second: *where is the ball*, and *how fast is it moving*? The answers must be good in three ways:

1. **Frequent.** The ball moves quickly, so the controller needs a fresh position roughly **50 times per second or more**. If it only learns the position a few times per second, the ball has rolled far away before the controller can react.
2. **Accurate.** The position should be correct to within **a few millimetres**. If the measured position jumps around, the controller will jerk the plate back and forth in response to noise.
3. **Timely.** The measurement should be as **fresh** as possible. A controller acting on where the ball *was* 100 milliseconds ago is always a step behind, and tends to overshoot.

### 2.2 How existing systems sense the ball

| Sensor | Advantages | Disadvantages |
|---|---|---|
| **Camera** above the plate | Accurate and well understood | Needs a mount above the plate, good lighting and image-processing software, and adds delay |
| **Resistive touchscreen** as the plate | Fast and simple | The plate *must* be the touchscreen, and the ball must be heavy enough to press it |
| **ToF distance sensor** at the plate edge (this project) | Tiny, cheap, mounts on the plate edge, no image processing | Measures only a *distance along one beam*, the beam is wide, and it gives at most about 100 readings per second |

### 2.3 The research gap

Our literature review found the following:

- **No closed-loop, two-dimensional ball-on-plate system using ToF sensors** has been published.
- ToF sensors (the VL53L0X) appear only in **ball-and-beam** projects. That is the one-dimensional version of the problem, where a ball rolls along a single tilting rail. Even there, nobody reports how accurate or how fast the sensing was.
- The closest related work is by **Bayar (2021)**, who used a spinning laser scanner on a commercial ball-on-plate rig. It was only used to *detect* the ball, not to control it, and it ran at just 10 updates per second. The author identified the scanner as the bottleneck.

So there is a genuine gap: nobody has shown that ToF sensing can be good enough to balance a ball in two dimensions. The full literature review is kept in the Claude Doc *"ToF Sensing for Ball-on-Plate Balancing: Systematic Review"*.

---

## 3. The sensor, and why it makes the problem hard

### 3.1 The VL53L4CD

The project uses the **ST VL53L4CD**, a small single-zone ToF sensor. Its key properties are:

| Property | Value |
|---|---|
| Maximum reading rate | 100 readings per second (each reading takes 10 ms) |
| Beam width | about 18 degrees; the beam spot is about 32 mm wide at 10 cm and 48 mm wide at 15 cm |
| Accuracy at short range | about ±7 to 8 mm |
| What it reports | a single distance, plus some quality numbers such as signal strength and an estimate of its own noise |

### 3.2 The core difficulty: distance without direction

A single-zone ToF sensor tells you **how far away** something is, but not **in which direction**. Its beam is a cone about 18 degrees wide. When it reports "something is 20 cm away", that something could be anywhere inside the cone at that distance: on the left edge of the cone, the right edge, or the middle. At 20 cm, an 18-degree cone is about 6 cm wide, larger than the ball itself.

To turn distances into a two-dimensional position, there are two basic strategies:

1. **Use several sensors and intersect their distances.** If sensor A says the ball is 20 cm away, the ball lies somewhere on a circle of radius 20 cm around A. If sensor B, at another corner, says 15 cm, the ball also lies on a circle of radius 15 cm around B. The ball must be where the two circles cross. This is called **trilateration**, and it is the same principle GPS uses.
2. **Sweep the beam.** If the sensor is mounted on a small motor that turns it, we know which way the beam is pointing at each moment. When the sensor sees the ball, the beam direction gives us the bearing and the distance gives us the range. Together they give a position.

### 3.3 How the design evolved

We worked through several designs, partly on paper and partly in simulation.

**Design 1: one sensor spinning in a full circle. Rejected.** A single sensor spinning around would sweep the whole plate, but it spends most of its time pointing in directions where the ball is not. If it takes one reading every 10 degrees, a full turn needs 36 readings, so even at 100 readings per second it sees the ball only about **2.8 times per second**. That is far too slow (Appendix A.8 has the calculation). We keep this as the *motivating negative result*: it shows clearly why naive scanning does not work.

**Design 2: four sensors, one in each corner, each sweeping its own quarter of the plate. This is the design currently simulated.**
- Each sensor sits in a corner and only needs to watch a 90-degree wedge: the quarter of the plate in front of it.
- It sweeps back and forth across that wedge in 18-degree steps, taking one reading per step, so one sweep takes 5 readings (50 ms).
- The four sensors are deliberately **phase-staggered**: they are shifted in time so that they don't all point in the same direction at once. Without this, opposite sensors shine their infrared light straight into each other and blind each other.
- On paper this arrangement gives about **80 position updates per second**, comfortably above the 50 we need.

**Design 3: fixed sensors that don't move. These are the fallback routes for hardware.**
- *Two fixed sensors at adjacent corners*, combined by trilateration. Simple and fast (up to 100 updates per second), but less accurate when the ball is near the line joining the two sensors.
- *One multi-zone VL53L5CX sensor* on one edge. This sensor has a 4×4 or 8×8 grid of zones, so the zone that sees the ball tells us its direction.

As section 6.2 will show, the simulation found that ordinary hobby servo motors are too slow to sweep the sensors as Design 2 requires. This makes the fixed-sensor routes more attractive for real hardware.

---

## 4. What the simulation contains

### 4.1 Layout

The picture below shows the simulated plate from above, and the platform from the side.

```
   Top view:

            sensor 3 (-,+)                    sensor 2 (+,+)
               ◢──────────────────────────────◣
               │                              │
               │         30 cm plate          │      ● = 40 mm ball
               │                              │
               │              ●               │      each corner sensor sweeps
               │                              │      its 90° quarter of the plate,
               │                              │      with the beam 2 cm above
               ◥──────────────────────────────◤      the plate surface
            sensor 0 (-,-)                    sensor 1 (+,-)

   Side view:

            plate ═══════════════════════════   (tilted by three servo legs)
                       /         |         \
                   servo A    servo B    servo C   (placed at 90°, 210° and 330°)
```

### 4.2 The parts

| Part | How it is simulated |
|---|---|
| **Plate** | A 30 × 30 cm square, 6 mm thick, matte black so it reflects little infrared light. In the learning environment it has **no rim**, so the ball can fall off the edge. |
| **Ball** | A 40 mm ping-pong ball weighing 2.7 g. It is hollow, which affects how it rolls, and white, so it reflects infrared light well. |
| **Platform** | A **3-RRS parallel platform**. Three servo motors sit under the plate, 120 degrees apart. Each turns a short arm (the *crank*), which pushes a rod up to a ball joint under the plate. Moving the three servos together raises and lowers the plate; moving them differently tilts it in any direction. |
| **Plate servos** | **MG996R** hobby servos by default. The simulation includes 19 real servo models from an online electronics shop, with their published speed and strength. |
| **Sensors** | Four VL53L4CD ToF sensors, one per corner. Each is mounted on its own small **SG90** servo that sweeps it back and forth. The beam sits 2 cm above the plate, at the height of the ball's middle (its *equator*). |
| **Ball tracker** | An **extended Kalman filter (EKF)**. This algorithm combines the noisy, scattered sensor readings with a physics prediction of how the ball moves, and produces a smooth estimate of the ball's position and velocity. |
| **Disturbances** | Random small "kicks" to the ball, and a slow, wandering push that represents things like air currents or an uneven plate. |

The physics is computed by **MuJoCo**, a widely used physics engine for robotics research. It advances the simulation in very small steps of 0.25 milliseconds, i.e. 4000 steps for every second of simulated time.

---

## 5. How the simulation works

This section explains each part of the simulation in words. The equations for every part are in [Appendix A](#appendix-a-the-mathematics).

### 5.1 The servo motors

A cheap **hobby servo** is a small motor with a gearbox and a built-in controller. You tell it what angle to go to, and it drives itself there. It is easy to treat a servo as a perfect device that instantly jumps to whatever angle you ask for. Real servos are not like that, so the simulation models their actual behaviour.

**How you command a servo.** A hobby servo is not told its target angle as a number. Instead, it receives an electrical pulse, and the *width* of the pulse encodes the angle. A pulse 1500 microseconds wide means "go to the middle position", 500 µs means one end of the travel, and 2500 µs the other end. In our setup, the middle position (1500 µs) is defined as **angle 0**.

**How often a servo listens.** A standard servo only accepts a new pulse **once every 20 milliseconds** (50 times per second). Whatever you command in between is ignored until the next 20 ms tick. Some newer "digital" servos accept updates up to 333 times per second.

**How fast and how strong.** Every servo has a maximum speed and a maximum torque (turning strength), listed by the manufacturer. Speed is usually given as "seconds per 60 degrees", for example 0.15 s/60°. Torque is given as "kilogram-centimetres", for example 11 kg·cm. Both depend on the supply voltage, so the simulation interpolates them for the chosen voltage.

**Imperfections.** Real servos also have:
- a **deadband**: a very small change in the command is ignored, because the servo's internal controller does not react to tiny errors;
- **backlash**: a little play in the gears, so the output shaft can wobble slightly even when the motor holds still;
- **friction** in the gears;
- **inertia**: the motor cannot speed up or slow down instantly.

The simulation includes all of these. Where a datasheet does not give a value (backlash is rarely published, for example), the simulation uses a reasonable estimate and marks it as **assumed** in the configuration file, so it is always clear which numbers are measured facts and which are educated guesses.

**Zero means flat.** We set up the platform geometry so that when all three servos are at angle 0, each crank arm is horizontal and the plate is perfectly flat. We verified this both mathematically (Appendix A.3) and in the simulation.

### 5.2 The ToF sensors

The simulation does not simply report the true distance with some random noise added. It models how the real sensor produces a reading:

1. **Casting light rays.** Every 2.5 milliseconds, each sensor sends out 300 simulated light rays spread across its beam cone. Each ray travels until it hits something (the ball, the plate surface, the far edge of the plate, or another sensor's housing). The simulation records what was hit, how far away it is, and how strongly that surface reflects infrared light. The white ball reflects strongly; the black plate weakly.
2. **Building a histogram.** The real sensor sorts the returning light by how long it took to come back, building a *histogram* of light intensity versus distance. The simulation does the same, in 2 mm distance bins.
3. **Picking the answer.** After 10 milliseconds (four batches of rays), the sensor looks at the histogram and picks the **strongest peak**, i.e. the distance from which the most light came back. It reports that distance. If the returned signal is weak, the reading is noisier, just as with the real sensor.
4. **Rejecting bad readings.** If the reading is too noisy, the sensor flags it as **invalid**, exactly as the real sensor's driver software does.
5. **Interference.** Infrared light from the *other* sensors can leak into each sensor's receiver, either reflected off surfaces or shining directly. The simulation adds this as extra background light, which makes readings noisier.

### 5.3 Estimating the ball's position: the Kalman filter

The raw sensor readings arrive at irregular times, from different sensors pointing in different directions. Some are noisy, some are invalid, and some see the background instead of the ball. The **extended Kalman filter (EKF)** turns this messy stream into one smooth, reliable estimate of where the ball is and how fast it is moving.

The EKF repeats two steps, *predict* and *correct*:

- **Predict.** Between readings, the filter moves its estimate forward using physics. It knows how much the plate is tilted, so it knows which way gravity is pulling the ball and how fast the ball should be speeding up. If the ball was at position X moving at speed V, a moment later it should be a bit further along. The filter also tracks its own **uncertainty**, which grows during prediction because the physics is never perfect.
- **Correct.** When a valid reading arrives, the filter compares it with what it expected that sensor to see. If the reading agrees, the estimate is nudged slightly toward it. If the reading disagrees, the filter weighs how much to trust the reading against how much to trust its own prediction, and moves the estimate by a sensible amount. Readings that are very precise move the estimate a lot; vague readings move it a little.

The filter also applies some extra rules:

- **Ignoring the background.** When the simulation starts, each sensor records what it sees on an *empty* plate, for example the distance to the far edge. Any later reading that matches this empty-plate distance is thrown away, because it is not the ball.
- **Rejecting outliers.** A reading that disagrees wildly with the prediction is treated as an error and ignored.
- **Starting and restarting.** The first good reading sets the initial position. If no good reading arrives for 0.3 seconds, the filter assumes it has lost the ball and starts again from the next reading.

An important consequence, explained with the mathematics in Appendix A.6: **one reading on its own places the ball somewhere along a thin arc, not at a single point.** The distance is quite precise, but the direction is vague because the beam is wide. Combining readings from several sensors, whose arcs cross at the ball, is what gives an accurate position.

### 5.4 What happens in one simulation step

Every 0.25 milliseconds of simulated time, the simulation does the following, in this order:

1. Each plate servo computes how much torque its motor produces, based on its command and its current angle.
2. Each scan servo does the same, which moves the sensors along their sweep.
3. The random disturbance forces are applied to the ball.
4. Every 2.5 ms the sensors cast their light rays, and every 10 ms each sensor finishes a reading.
5. The Kalman filter predicts forward, and corrects itself when a new reading has arrived.
6. MuJoCo advances the physics of the plate, ball and servos by one small step.

---

## 6. Results so far

### 6.1 How well can the sensors track the ball?

In these first experiments there is no balancing controller. Instead, the plate follows a **scripted pattern of tilts**: holding still, tilting back and forth, tilting in circles, or tilting randomly. The ball rolls around in response, and we measure how far the estimated position is from the true position. We report this tracking error as a **root-mean-square error (RMSE)** in millimetres. Roughly speaking, this is the typical size of the error.

**Effect of the number of sensors.** This experiment used 330 simulated runs over 11 different scenarios, with strong disturbances.

| Number of sensors | Average error | Error in the worst scenario |
|---|---|---|
| **4 (one per corner)** | **13.5 mm** | 30.7 mm |
| 3 | 15.6 mm | 39.1 mm |
| 2 at adjacent corners | 17.7 mm | 34.9 mm |
| 2 at opposite corners | 17.6 mm | 36.2 mm |
| 1 | 29.1 mm | 62.5 mm |

More sensors clearly help. A single sensor is about twice as bad as four, because one sensor can only ever say "the ball is somewhere along this arc".

**What else matters, and what doesn't** (with four sensors):

- **Staggering the sweeps matters.** If all four sensors sweep in step, opposite sensors shine into each other. 14% of readings then become invalid, and the error rises to 14.5 mm.
- **Larger sweep steps are better.** Sweeping in 18-degree steps gave 13.5 mm, against 14.7 mm for 10-degree steps. With larger steps a sweep takes fewer readings, so each sensor passes over the ball more often.
- **Many details hardly matter.** Moving the sensor continuously or stopping it for each reading, switching interference on or off, and modelling the beam width exactly or approximately all changed the result by less than 1 mm.
- **Calm versus violent conditions.** With gentle motion the error is 4 to 12 mm. With violent motion (the ball moving at up to 0.8 metres per second, with hard kicks) it rises to 17 to 31 mm.
- **The weakest spot** is when the ball sits close to the straight line between two sensors. There the two distance circles cross at a very shallow angle, so a small error in distance becomes a large error in position (13.5 mm in the held-ball test; see Appendix A.6 for why).
- **Detection rate.** When the beam is actually pointing at the ball, the sensor detects it in 53% to 85% of readings. A small ball far away does not return much light in a 10 ms reading, so it is sometimes missed.

**Effect of the servo-driven plate.** With real servo motors tilting the plate, the plate follows the commanded tilt to within 0.0 to 0.8 degrees. The 50-updates-per-second limit of standard servos costs about 0.15 degrees of accuracy. Faster (333 Hz) digital servos help slightly, but stronger servos do not help, because the plate is light and torque is never the limiting factor.

### 6.2 An important finding: hobby servos cannot sweep the sensors fast enough

Design 2 requires each sensor to sweep 90 degrees in 50 milliseconds. That is a turning speed of about **1800 degrees per second**, with an instant reversal of direction at each end. Typical hobby servos manage only about **400 to 600 degrees per second**, and they cannot reverse instantly.

When we simulated the sweep with real servo models, the servos could not keep up. Instead of sweeping across the whole 90 degrees, they only wiggled back and forth over a small range near the middle:

| Scan servo | Commanded sweep | Sweep actually achieved |
|---|---|---|
| SG90 | from −45° to +45° | from −33° to −21° |
| MG945 | from −45° to +45° | from −12° to −5° |
| DS3218 (at 333 Hz updates) | from −45° to +45° | from −3° to +4° |

In other words, **with real hobby servos the "sweeping" sensors are in effect fixed**, pointing roughly at the middle of their quarter of the plate. Interestingly, the SG90, the smallest and cheapest servo, does best, because it is the lightest and fastest.

What this means for tracking:

| Case | Tracking error |
|---|---|
| A perfect sweep, as a stepper motor or galvanometer mirror could provide, in the circle scenario | 6.9 mm |
| The SG90 sweep, same circle scenario | 27 mm |
| The ball held near the centre by a balancing controller, with either kind of sweep | about 6 mm |

The poor sweep mostly hurts when the ball wanders far from the centre, where the nearly fixed beams cannot see it well. This finding favours one of three options for real hardware: a faster sweeping mechanism such as a stepper motor or a mirror, a slower sweep with fewer steps, or simply **fixed sensors** (Design 3).

### 6.3 Balancing the ball

With a controller in the loop, the simulation now actually tries to keep the ball at the centre.

| Controller | Result |
|---|---|
| **Hand-tuned PID controller**, using the Kalman filter's estimate | Never dropped the ball, in 4 out of 4 test runs of 10 seconds each. It keeps the ball within about 26 to 29 mm of the centre on average. |
| **No control at all** (plate held flat) | The ball rolls off the plate in about 2 seconds, because of the random disturbances. |
| **Neural network trained by reinforcement learning** | The training system is complete and tested, but **no full training run has been done yet** (see section 8). |

### 6.4 Checking that the physics is right

A simulation is only useful if it behaves like the real world, so we checked:

- **Rolling.** A ball on a plate held at a 2-degree tilt accelerates within 1% of the value predicted by textbook physics.
- **Servo speed.** When asked to tilt the plate by 8 degrees, the simulated platform reaches 90% of that tilt in 80 to 165 milliseconds, depending on the servo model. This is plausible given the servos' published speeds.
- **Sensor accuracy.** The simulated sensor's noise and maximum range match the tables in the VL53L4CD datasheet.

---

## 7. The learning environment

To let a neural network learn to balance the ball, we wrapped the simulation in a standard **Gymnasium** interface. Gymnasium is the common format for reinforcement-learning environments in Python, so almost any learning library can use ours. The environment is called `BallOnPlateToF-v0`.

Reinforcement learning works by trial and error. The learning program (the **agent**) repeatedly looks at the current situation (the **observation**), chooses what to do (the **action**), and receives a score (the **reward**) that says how well it is doing. Over many thousands of attempts it learns which actions lead to high rewards.

### 7.1 What the agent controls: the action

The agent outputs three numbers, one for each plate servo, each between −1 and +1:
- **−1 and +1** mean the servo turns 80 degrees one way or the other;
- **0** means the servo is at its middle position.

All three at 0 means the plate is flat. Just like a real servo, the command takes effect at the next 20 ms update.

### 7.2 What the agent sees: the observation

| Observation | What it contains |
|---|---|
| `tof` | The **raw readings** from every sensor: the last few readings from each, with whether the reading is valid, the measured distance, the sensor's own noise estimate, the signal strength, the background light level, which way the beam was pointing, and how old the reading is |
| `ball_est` | The Kalman filter's estimate of the ball's position and velocity, plus a flag saying whether the filter currently has a lock on the ball |
| `servo_cmd` | The command the agent sent last time |
| `target` | Where the ball should be (by default the centre) |
| *optional extras* | The servos' measured angles (only for servo models with a position-feedback wire), and the plate's measured tilt (as an accelerometer would give) |

### 7.3 How the agent is scored: the reward

Every 20 ms the agent receives a score made of several parts:

- **+1** simply for keeping the ball on the plate;
- **minus** an amount that grows with the ball's distance from the target, so being far from the centre is penalised;
- **minus** a small amount for the ball's speed, so a still ball is preferred;
- **minus** a small amount for sudden changes in the servo commands, so smooth control is preferred.

**If the ball falls off the edge, the agent receives −20 and the attempt (the *episode*) ends immediately.** Otherwise an episode lasts 10 seconds, i.e. 500 decisions.

### 7.4 Everything is configurable

Every parameter of the simulation lives in plain-text configuration files in the `config/` folder: plate size, servo model, sensor positions, noise levels, reward weights and so on. Any of them can be changed when creating the environment, without touching the code. For example, you can ask for a different servo model, a perfect sweep instead of servo-driven sweeping, or only two sensors.

Parameters can also be **randomised at the start of every episode**. For example, the gear backlash can be drawn randomly from a range each time. This trains a network that copes with the variation it will meet on real hardware.

A detailed technical reference for the environment is in `docs/ENV.md`.

---

## 8. The neural network and how it is trained

### 8.1 The network

The network we designed for this task (in `control/tof_net.py`) works as follows:

- **Input.** It looks only at the **raw ToF readings**: 4 sensors × the last 10 readings from each × 7 numbers per reading, i.e. 280 numbers in total. It does not see the Kalman filter's estimate. The idea is to find out whether a network can learn to balance directly from raw sensor data.
- **Body.** Three layers of 256 artificial neurons each. Every layer uses two well-known techniques:
  - **layer normalisation** rescales each layer's values to a consistent range, which keeps training stable;
  - **dropout** switches off a random 1% of the neurons at each training step, which prevents the network from relying too heavily on any single neuron.
- **Output.** Three numbers: the commands for the three plate servos.

### 8.2 The training method: Soft Actor-Critic (SAC)

We train the network with **Soft Actor-Critic (SAC)**, a modern reinforcement-learning algorithm. It suits this task for two reasons:

1. It is designed for **continuous** actions, like servo angles that can take any value, rather than a fixed menu of choices.
2. It **reuses past experience**. Every attempt is stored in a memory (the *replay buffer*) and replayed many times during learning. This matters because our detailed simulation is slow, so every simulated second is valuable.

**Hardware use.** Fourteen copies of the simulation run in parallel on the computer's processor, generating experience at about 130 to 210 decisions per second in total. Meanwhile the network itself is trained on the graphics card (an NVIDIA RTX 3070). One million training steps take roughly **two hours**.

### 8.3 Training conveniences

| Feature | What it does |
|---|---|
| **Weights & Biases (W&B) dashboard** | Shows the training progress live in a web browser: the reward, how often the ball falls off, how far the ball ends up from the target, and the network's internal learning statistics. The secret login key is stored in a private `.env` file that is never uploaded to GitHub. |
| **Checkpoints** | A copy of the network is saved every 200,000 steps. Only the five most recent copies are kept, to save disk space. |
| **Best model** | A separate file, `best.zip`, is updated whenever the network's average score over its last 100 attempts beats its previous best. |
| **Resuming** | Training can be stopped and continued later from the final, best or any saved network. Pressing Ctrl-C saves everything before stopping. |
| **Alternatives** | Four other algorithms (TD3, DDPG, PPO and A2C) are also supported, as is a standard network that sees all observations rather than just the raw readings. |

---

## 9. How to run everything

All commands are run from the project folder in a terminal.

### 9.1 Installation (once)

The project needs Python 3.13 and the `uv` package manager.

```bash
uv venv .venv -p 3.13
uv pip install --python .venv/bin/python -r requirements.txt
```

The first command creates a private Python environment in the `.venv` folder. The second installs everything the project needs into it: the physics engine, the learning libraries and the project itself.

If the installation **times out** while downloading the large NVIDIA graphics libraries, run it with a longer timeout:

```bash
UV_HTTP_TIMEOUT=900 UV_CONCURRENT_DOWNLOADS=2 uv pip install --python .venv/bin/python -r requirements.txt
```

On a computer **without an NVIDIA graphics card**, install the processor-only version of PyTorch instead. `README.md` gives the exact commands.

### 9.2 Setting up the training dashboard (once)

Copy the template file and paste your Weights & Biases key into it:

```bash
cp .env.example .env
# then open .env in an editor and fill in:  WANDB_API_KEY=your-key-here
```

### 9.3 Training

To train the raw-ToF network with SAC for one million steps:

```bash
.venv/bin/python control/train.py --algo sac --net tof --n-envs 14 --steps 1000000 --out runs/sac_tof_1m
```

The results are saved in the folder `runs/sac_tof_1m`. To continue that training later for another 500,000 steps:

```bash
.venv/bin/python control/train.py --resume runs/sac_tof_1m/final.zip --steps 500000
```

### 9.4 Testing a controller

To test the best trained network over 10 episodes:

```bash
.venv/bin/python control/evaluate.py --controller model --path runs/sac_tof_1m/best.zip --episodes 10
```

To test the hand-tuned PID controller instead:

```bash
.venv/bin/python control/evaluate.py --controller pid --episodes 10
```

To **watch** a controller balancing the ball in a live 3-D window, add `--view`:

```bash
.venv/bin/python control/evaluate.py --controller pid --view
```

### 9.5 Changing settings from the command line

Any configuration value can be changed with `--set`. Some useful examples:

- `--set scan.actuator=ideal` uses a perfect sensor sweep instead of the SG90 servos;
- `--set platform.servo_model=DS3218` uses a different servo for the plate;
- `--set scan.corners=[0,1]` uses only two sensors, at corners 0 and 1.

### 9.6 Running the simulation studies

These run the scripted experiments of section 6.1, without a balancing controller:

```bash
.venv/bin/python tools/calibrate.py        # checks the physics and sensor model against theory and the datasheet
.venv/bin/python tools/run_scenarios.py    # runs all scripted scenarios; plots go to sim/out/
.venv/bin/python tools/ablate.py           # the large comparison of sensor and servo variants
```

---

## 10. Where things are in the code

| Folder or file | What it contains |
|---|---|
| `bop_env/` | The simulation and the learning environment. This folder deliberately contains no learning code, so the environment can be used with any learning library. |
| `control/` | The PID controller, the neural network, the training script and the evaluation script |
| `tools/` | Scripts for the scripted experiments, the large comparisons, the physics checks and the plots |
| `config/` | Every parameter, one file per part of the system: plate, sensor, servos, scan pattern, tracker, environment, controllers |
| `docs/ENV.md` | Detailed technical reference for the learning environment |
| `docs/STATUS.md` | This document |
| `README.md` | Installation and run instructions |
| `sim/out/` | Plots and summary tables from the experiments |
| `runs/` | Training results (kept on the computer, not uploaded to GitHub) |

---

## 11. Project status

### 11.1 What is finished ✅

- The **literature review**, which identified the research gap.
- The **feasibility analysis**, which rejected the single spinning sensor.
- The **complete physics simulation**: plate, ball and three-servo platform; 19 real servo models with their published specifications; ray-traced ToF sensors; the Kalman filter tracker; and random disturbances.
- **Tracking experiments** comparing the number of sensors, sweep settings, interference and servo models, about 1000 simulated runs in total.
- A **PID controller** that successfully balances the ball.
- The **learning environment**, with raw sensor observations, an episode that ends when the ball falls off, and full configurability.
- The **neural network and training system**, with the live dashboard, checkpoints, best-model saving and resuming.
- The code is **on GitHub**, and everything can be installed with one command.

### 11.2 What is not done yet ⏳

- **A full training run of the neural network**, and a comparison of the trained network against the PID controller.
- **Re-running the older tracking experiments** with the newest, more realistic servo model. Most results in section 6.1 were produced before that model existed.
- **Running the physics on the graphics card.** At the moment only the neural network uses it; the physics runs on the processor.
- **Confirming the research gap** with a formal search of the Scopus and IEEE Xplore databases. So far only a web search has been done, so "nobody has done this" strictly means "we did not find it".
- **Any real hardware.**

### 11.3 Known limitations ⚠️

- **The hobby scan servos are too slow for the planned sweep** (section 6.2). With the default settings the sensors are almost stationary.
- **The network sees only the raw sensor readings, not its own previous command.** It therefore has no direct way to know how the plate is currently tilted, and must work it out indirectly from how the ball is accelerating. Adding the previous command to its input is a small change and is recommended before a long training run.
- **Some servo properties are educated guesses**: backlash, friction and response time are not published by the manufacturers.
- The simulation leaves out a few real-world effects: the supply voltage does not drop when the servos draw a lot of current, there is no accelerometer noise, and the ball and plate are geometrically perfect.
- **By default the reward uses the ball's true position**, which is known in simulation but would not be on real hardware. An option to use the estimated position instead is available.

### 11.4 Suggested next steps

1. Add the previous servo command to the network's input.
2. Run a full SAC training of one to two million steps, once with a perfect sweep and once with the SG90 servos, and compare both against the PID controller.
3. Decide the hardware route: fixed corner sensors, a faster sweep mechanism, or the multi-zone VL53L5CX.
4. Build the physical plate, and measure the real sensors' error and missed-detection rate on balls of 30 to 60 mm with different surface finishes.
5. Compare ToF sensing against a touchscreen and a camera on the same plate with the same controller.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **Time-of-flight (ToF)** | A way of measuring distance by timing how long a pulse of light takes to bounce back from an object |
| **Trilateration** | Finding a position from distances to several known points, by finding where the circles of those distances cross |
| **Phase stagger** | Shifting the timing of the sensors' sweeps so that they never point at each other at the same moment |
| **Kalman filter (EKF)** | An algorithm that combines a physics prediction with noisy measurements to produce a smooth, reliable estimate. "Extended" means it handles non-linear measurements such as distances and angles. |
| **RMSE** | Root-mean-square error: a standard way of summarising the typical size of an error |
| **3-RRS platform** | A platform held up by three legs, each made of a rotating servo arm (R), a rod with a hinge (R) and a ball joint (S) |
| **PWM pulse** | The electrical signal used to command a hobby servo; the pulse width sets the target angle |
| **Deadband** | The range of small command changes that a servo ignores |
| **Backlash** | Play, or looseness, in a gearbox |
| **PID controller** | A classic controller that reacts to the error (P), how the error builds up over time (I), and how fast the error is changing (D) |
| **Reinforcement learning** | A way of training a program by trial and error, rewarding good outcomes |
| **Gymnasium** | The standard Python format for reinforcement-learning environments |
| **SAC (Soft Actor-Critic)** | A reinforcement-learning algorithm suited to continuous actions such as motor angles |
| **Episode** | One attempt at the task: 10 seconds, or until the ball falls off |
| **Checkpoint** | A saved copy of the network part-way through training |
| **W&B (Weights & Biases)** | An online dashboard for watching training progress |

---

## Appendix A. The mathematics

This appendix gives the equations behind every part of the simulation, the position estimate, and the sweep-rate calculations. Each formula matches the code exactly, and the relevant code file is named in each section heading.

**Coordinates.** All positions are measured in the **plate frame**: a coordinate system attached to the plate, which tilts with it. Its origin is at the centre of the plate's top surface, the $x$ and $y$ axes run parallel to the plate edges, and $z$ points up out of the plate.

### A.1 How the ball moves

The physics engine MuJoCo computes the full motion of the ball, plate and servos, including all the forces where they touch. It uses an integrator called `implicitfast` with a time step of $\Delta t = 0.25\,\text{ms}$, and models friction in all directions, including resistance to rolling and to spinning in place.

For other parts of the system (the Kalman filter, the PID controller, and checking the physics), we also need a simple formula for how fast the ball accelerates on a tilted plate. Consider a ball rolling without slipping on a slope of angle $\theta$. Part of gravity's pull goes into moving the ball forward and part into making it spin. How the pull divides depends on the ball's **moment of inertia** $I$, a measure of how hard the ball is to spin up. The acceleration down the slope is:

$$
a = \frac{g \sin\theta}{1 + \dfrac{I}{m r^2}}
$$

where $g = 9.81\,\text{m/s}^2$ is gravity, $m$ is the ball's mass and $r$ its radius. A ping-pong ball is a hollow shell, for which $I = \tfrac{2}{3} m r^2$. Substituting this gives:

$$
a = \frac{g\sin\theta}{1 + \tfrac{2}{3}} = \tfrac{3}{5}\, g \sin\theta \approx 0.6\; g\sin\theta
$$

So a hollow ball accelerates at only 60% of the rate a frictionless sliding block would; the other 40% of gravity's effort goes into spinning it. We call this factor $k = 0.6$, the **roll factor**. For comparison, a solid ball would have $k = 5/7 \approx 0.71$.

In three dimensions, with the plate's orientation described by a rotation matrix $R$, the ball's acceleration in the plate frame is the in-plate part of gravity, scaled by $k$:

$$
\mathbf{a}_{xy} = k \,\big(R^{\mathsf T}\mathbf{g}\big)_{xy}, \qquad \mathbf g = (0,\,0,\,-9.81)
$$

Here $R^{\mathsf T}\mathbf g$ expresses gravity in the plate's own coordinates, and the subscript $xy$ keeps only the part that lies along the plate surface. We checked the simulation against this formula: on a plate held at a 2° tilt, the simulated ball's acceleration agrees with it to within 1%.

**Bouncing.** The contact between ball and plate is slightly springy, with a time constant of 10 ms and a damping ratio of 0.05. This makes a dropped ball bounce back with about 85% of its impact speed, which is realistic for a ping-pong ball.

**When does the ball fall off?** At every physics step, the simulation checks whether the centre of the ball has passed the edge of the plate, or has dropped below the plate surface:

$$
\max(|x|,\,|y|) > \tfrac{L_p}{2} \quad\text{or}\quad z < 0
$$

where $L_p = 0.30\,\text{m}$ is the plate width and $(x, y, z)$ is the ball's centre in the plate frame. Once the centre is past the edge, the ball tips over and falls. The episode then ends with a reward of $-20$.

### A.2 Random disturbances (`bop_env/disturb.py`)

To make the task realistic, two kinds of random horizontal force $\mathbf f(t)$ act on the ball.

**Kicks** are short, sharp pushes, like someone flicking the ball:
- The number of kicks in an episode of length $T$ follows a Poisson distribution, the standard model for random events in time: $N \sim \text{Poisson}(\lambda T)$, with an average rate of $\lambda = 0.2$ kicks per second.
- Each kick has a random strength between 50% and 150% of $F_k = 0.03\,\text{N}$, a random direction, and lasts 20 milliseconds.

**The slow push** represents gentle, wandering forces such as air movement or a slightly uneven plate. It is modelled as an **Ornstein–Uhlenbeck process**: a random force that drifts smoothly and keeps being pulled back towards zero. At each time step it is updated as:

$$
\mathbf{f}_{t+\Delta t} = \mathbf{f}_t \;-\; \frac{\Delta t}{\tau}\,\mathbf{f}_t \;+\; \sigma\sqrt{\frac{2\Delta t}{\tau}}\;\boldsymbol{\xi}, \qquad \boldsymbol{\xi}\sim\mathcal N(0, I_2)
$$

The first term keeps the current force, the second slowly decays it towards zero, and the third adds a small random nudge ($\boldsymbol\xi$ is a pair of standard random numbers). The parameter $\tau = 0.3\,\text{s}$ sets how quickly the force wanders, and $\sigma = 0.3\,\text{mN}$ sets its typical size.

### A.3 Platform geometry: from desired tilt to servo angles (`bop_env/legs.py`)

To tilt the plate to a particular angle, we need to know what angle each of the three servos must turn to. This is the **inverse kinematics** problem. For this platform it has an exact, closed-form solution.

**The geometry.** Leg $i$ is placed at angle $\alpha_i$ around the centre, where $\alpha_i \in \{90°, 210°, 330°\}$. Each leg consists of:
- a servo whose rotation axis sits at $\mathbf{B}_i = (r_b\cos\alpha_i,\; r_b\sin\alpha_i,\; h_s)$;
- a crank arm of length $a$ attached to the servo;
- a rod of length $L$ from the end of the crank up to a ball joint under the plate;
- the ball joint itself, at $\mathbf{p}_i = (r_a\cos\alpha_i,\; r_a\sin\alpha_i,\; -t)$ in plate coordinates.

The dimensions are: $r_b = 90$ mm (servo distance from centre), $r_a = 120$ mm (joint distance from centre), $a = 30$ mm (crank), $L = 80$ mm (rod), $h_s = 40$ mm (servo height) and $t = 6$ mm (plate thickness).

**The desired plate pose.** We want the plate tilted by a roll angle $\phi$ (about the $x$ axis), a pitch angle $\vartheta$ (about the $y$ axis), and raised by a height $h$. Its orientation and centre position are then:

$$
R = R_x(\phi)\,R_y(\vartheta), \qquad \mathbf{c} = \big(0,\,0,\,z_0 + h\big), \qquad z_0 = h_s + L + t
$$

Here $z_0$ is the plate's normal resting height.

**Where the ball joint ends up.** We work out the position of leg $i$'s ball joint, then express it in that leg's own coordinate frame, which is rotated by $\alpha_i$ so that the crank swings in the leg's local $x$–$z$ plane:

$$
\mathbf{P} = R_{l,i}^{\mathsf T}\big(\mathbf{c} + R\,\mathbf{p}_i - \mathbf{B}_i\big)
$$

**Solving for the servo angle.** When the servo is at angle $q$, the tip of the crank is at $\mathbf{C}(q) = a\,(\cos q,\; 0,\; -\sin q)$. The rod connects the crank tip to the ball joint and has fixed length $L$, so:

$$
\lVert \mathbf{P}-\mathbf{C}(q)\rVert = L
$$

Squaring both sides and expanding gives an equation with $q$ appearing only inside a cosine and a sine:

$$
P_x\cos q - P_z \sin q = g, \qquad \text{where } g = \frac{\lVert\mathbf P\rVert^2 + a^2 - L^2}{2a}
$$

To solve this, we combine the cosine and sine into one shifted cosine. Write $P_x = n\cos\beta$ and $-P_z = n\sin\beta$, where $n = \sqrt{P_x^2+P_z^2}$ and $\beta = \operatorname{atan2}(-P_z,\, P_x)$. The equation becomes:

$$
n\cos(q-\beta) = g \quad\Longrightarrow\quad q = \beta \pm \arccos\!\left(\frac{g}{n}\right)
$$

There are two mathematical solutions, corresponding to the crank pointing up or down. We choose the one closer to $q = 0$, and limit it to the servo's range of $\pm 80°$.

**Proof that zero means flat.** The geometry is designed so that $r_a = r_b + a$. When the plate is flat at its normal height, the ball joint sits directly above the crank tip, so $\mathbf P = (a,\, 0,\, L)$. Then $g = a$, $n = \sqrt{a^2+L^2}$ and $\beta = -\arctan(L/a)$. Substituting:

$$
q = -\arctan\frac{L}{a} + \arccos\frac{a}{\sqrt{a^2+L^2}} = -\arctan\frac{L}{a} + \arctan\frac{L}{a} = 0
$$

So all three servos at zero gives exactly a flat plate.

**Reading the plate's tilt back.** The plate's actual roll and pitch can be read from its rotation matrix $R$:

$$
\vartheta = \arcsin R_{13}, \qquad \phi = \operatorname{atan2}(-R_{23},\; R_{33})
$$

### A.4 The servo motor model (`bop_env/servo.py`)

This section follows a command from the controller, through the servo's electronics, to the torque its motor produces.

**Step 1: angle to pulse width.** A servo with total travel $\Theta$ maps the pulse range 500–2500 µs onto that travel, so each radian of angle corresponds to $k_{us} = 2000\,\mu\text{s}/\Theta$ microseconds of pulse. A command $q$ (in radians) becomes the pulse:

$$
u = \operatorname{clip}\!\Big(\Delta_{pwm}\,\operatorname{round}\!\big(\tfrac{1500 + k_{us}\,q}{\Delta_{pwm}}\big),\; 500,\; 2500\Big)\;\mu\text{s}
$$

The rounding reflects that the controller's pulse generator can only produce pulse widths in steps of $\Delta_{pwm}$ (1 µs here). The clipping keeps the pulse in the valid range.

**Step 2: the servo only listens at fixed times.** The servo accepts a new pulse only at the start of each frame, at times $t_k = k / f_{fr}$. The frame rate is $f_{fr} = \min(f_{host}, f_{max})$: the slower of what the controller sends and what the servo accepts, normally 50 Hz. Between frames, the last accepted pulse $u_{lat}$ is held.

Optionally, the controller can limit how fast it lets the commanded angle change, which is the only way to slow a hobby servo down:

$$
\lvert u_{lat,k} - u_{lat,k-1}\rvert \le \omega_{host}\,k_{us}/f_{fr}
$$

**Step 3: the servo's internal controller.** Inside the servo, a potentiometer measures the angle of the output shaft (the *horn*). Because of gear backlash, the horn angle $q_h$ is the motor angle $q_m$ plus a small amount of play $q_\ell$: $q_h = q_m + q_\ell$. The servo compares the commanded pulse with the measured angle, also expressed in microseconds:

$$
e = u_{lat} - \big(1500 + k_{us}\,q_h\big)
$$

It then decides how hard to drive the motor. This drive level $D$, called the **duty**, runs from $-1$ (full power one way) to $+1$ (full power the other way). Small errors inside the deadband give no drive at all. Larger errors give a drive proportional to the error, up to full power once the error reaches a threshold angle $\theta_{pb}$ (the *proportional band*, assumed to be 4°):

$$
D = \begin{cases} 0 & \text{if } |e| \le \tfrac{1}{2}\,\text{deadband} \\[4pt] \operatorname{clip}\!\left(\dfrac{e}{k_{us}\,\theta_{pb}},\,-1,\,1\right) & \text{otherwise}\end{cases}
$$

**Step 4: the motor's torque.** A small DC motor produces its maximum torque when stalled, and less torque the faster it spins, reaching zero at its top speed. This straight-line relationship is:

$$
\tau = T_s\left(D - \frac{\omega_m}{\omega_0}\right)
$$

where $\omega_m$ is the motor's current speed, $T_s$ is the stall torque, and $\omega_0$ is the top (no-load) speed. Both come from the datasheet:

$$
\omega_0 = \frac{\pi/3}{t_{60}}, \qquad T_s = 0.0980665 \times \tau_{kg\cdot cm}
$$

$t_{60}$ is the listed time to turn 60° ($\pi/3$ radians), and $\tau_{kg\cdot cm}$ is the listed torque in kilogram-centimetres, converted to newton-metres. Both are interpolated for the chosen supply voltage.

**Step 5: inertia, friction and backlash.**
- The motor and gearbox have **inertia**, so they cannot change speed instantly. We choose the inertia $J$ so that the unloaded motor reaches about 63% of top speed in a time $\tau_m$ (assumed 20 ms). Starting from rest at full drive, the motor obeys $J\dot\omega = T_s(1 - \omega/\omega_0)$, which has time constant $J\omega_0/T_s$. Setting this equal to $\tau_m$ gives:

$$
J = \frac{\tau_m\, T_s}{\omega_0}
$$

- **Gear friction** is a constant opposing torque of $\tau_f = f \times T_s$, with $f = 2\%$ assumed.
- **Backlash** is modelled as a loose joint between the motor and the horn, free to move within $|q_\ell| \le b/2$ and stopped by stiff end stops.

**A worked example: the MG996R at 6.0 volts.**
- The datasheet gives 0.19 s/60° at 4.8 V and 0.15 s/60° at 6.6 V. Interpolating to 6.0 V gives $t_{60} = 0.163$ s, so the top speed is $\omega_0 = 6.41$ rad/s, or 367° per second.
- The torque, interpolated the same way, is 10.5 kg·cm, so $T_s = 1.03$ N·m.
- The pulse scale is $k_{us} = 636.6$ µs per radian, i.e. 11.1 µs per degree. Full drive is reached at a 4° error, i.e. 44 µs.
- The motor inertia is $J = 3.2\times10^{-3}$ kg·m², and the friction torque is 0.02 N·m.

### A.5 The ToF sensor model (`bop_env/tof.py`)

**Casting the rays.** Every 2.5 ms, each sensor casts $N = 300$ rays spread over a cone of half-angle $\theta_{max} = 11°$. To cover the cone evenly, the rays are spaced so that each covers the same solid angle (the same share of the cone's area), and they are arranged on a "golden-angle" spiral that avoids clumping. Ray $i$ points at angle $\theta_i$ from the beam axis and angle $\varphi_i$ around it:

$$
\cos\theta_i = 1 - \big(1-\cos\theta_{max}\big)\frac{i+\tfrac12}{N}, \qquad \varphi_i = i\,\pi\big(3-\sqrt5\big), \qquad \Omega_i = \frac{2\pi(1-\cos\theta_{max})}{N}
$$

where $\Omega_i$ is the solid angle each ray represents.

**How bright each ray is.** The sensor's light source (a *VCSEL*, a type of tiny laser) is brightest along the beam axis and fades towards the edges, following a Gaussian (bell-shaped) profile. Its half-width is $w = 8°$, measured where the intensity has fallen to $1/e^2 \approx 13.5\%$. The intensity at angle $\theta$, scaled so that the total emitted power is 1, and the power carried by ray $i$ are:

$$
I(\theta) = \frac{e^{-2\theta^2/w^2}}{\pi w^2/2}, \qquad P_i = I(\theta_i)\,\Omega_i
$$

**How much light comes back.** When ray $i$ hits a surface at distance $d_i$, the amount of light reflected back depends on:
- the surface's reflectivity, or **albedo**, $\rho$: 0.85 for the white ball, 0.10 for the black plate;
- the angle $\beta_i$ at which the ray hits the surface, since light hitting at a glancing angle is spread thinly;
- the distance, since the returning light spreads out and weakens with the square of the distance.

For a matte surface this gives the returned power:

$$
s_i = \frac{P_i\,\rho\,\cos\beta_i}{\pi\, d_i^2}
$$

The returning light is only counted if it falls within the receiver's field of view, which, according to the datasheet, narrows slightly with distance:

$$
\theta_i \le \theta_{rx}(d_i), \qquad \theta_{rx} \text{ going linearly from } 11° \text{ at } 0.1\,\text{m} \text{ to } 9° \text{ at } 1.0\,\text{m}
$$

**Building the histogram.** The returns are sorted into distance bins of width $\delta = 2\,\text{mm}$, and averaged over the $n = 4$ batches of rays that make up one 10 ms reading:

$$
h_j = \frac{1}{n}\sum_{\text{batches}}\;\sum_{\text{rays in bin } j} s_i
$$

**Finding the distance.** The sensor reports the distance of the strongest return:
1. The histogram is smoothed with a bell-shaped kernel of width $\sigma_p = 20\,\text{mm}$, matching the length of the real light pulse.
2. The highest point of the smoothed histogram, bin $j^*$, is found.
3. In a window $W$ around that peak (two pulse widths either side), the total signal $S$ is added up, and its centre of mass gives the measured distance $\hat r$:

$$
S = \sum_{j\in W} h_j, \qquad \hat r = \frac{1}{S}\sum_{j\in W} h_j\,c_j, \qquad c_j = (j+\tfrac12)\,\delta
$$

where $c_j$ is the distance at the centre of bin $j$.

**How noisy the reading is.** The datasheet gives the noise for one reference case: $\sigma_{ref} = 2.5\,\text{mm}$ for a white target at 200 mm, measured over $T_{ref} = 33\,\text{ms}$. The simulation scales this to other situations:
- a **weaker signal** gives more noise;
- a **shorter measuring time** gives more noise;
- more **background light** gives more noise.

$$
\sigma = \sigma_{ref}\,\sqrt{\frac{S_{ref}}{S}}\;\sqrt{\frac{T_{ref}}{T}}\;\sqrt{1 + \frac{A_{pk}}{S}}, \qquad A_{pk} = A\,\frac{|W|\,\delta}{1.3\,\text{m}}
$$

Here $S_{ref}$ is the signal the simulation computes at start-up for the datasheet's reference target, $T = 10\,\text{ms}$ is our reading time, and $A_{pk}$ is the part of the background light $A$ that falls inside the peak window. Background light is assumed to spread evenly over the sensor's 1.3 m range.

**The final reading.** The reported distance is the measured distance plus two kinds of error. One is fresh random noise in every reading. The other is a small fixed offset $o$ that is different for each sensor and stays the same for a whole episode, like a slight calibration error in a real sensor:

$$
r = \hat r + o + \varepsilon, \qquad \varepsilon\sim\mathcal N(0,\sigma^2), \qquad o\sim\mathcal N\big(0,\,(2\,\text{mm})^2\big)
$$

The reading is marked **valid** only if $\sigma \le 15\,\text{mm}$, the same threshold the real sensor's driver uses by default.

**Interference from other sensors.** Another sensor's light can reach this sensor's receiver in two ways, and both are added to the background light $A$:

- *Reflected:* the other sensor's light lights up spots on the ball or plate at points $\mathbf q$. If such a spot is inside this receiver's view and nothing blocks the line of sight, its reflected light is counted:

$$
A_{diff} = \sum_{\text{spots}} \frac{P\,\rho\,\cos\beta_{out}}{\pi\,\lVert\mathbf q - \mathbf o_{rx}\rVert^2}
$$

- *Direct:* if the two sensors face each other, each inside the other's cone, with a clear line of sight between them at distance $d$, the other sensor's light shines straight in:

$$
A_{dir} = \frac{I(\theta_{tx})}{d^2}
$$

### A.6 Estimating the ball's position: the Kalman filter (`bop_env/tracker.py`)

**What the filter tracks.** The filter's **state** is the ball's position and velocity in the plate frame:

$$
\mathbf x = \begin{pmatrix} x & y & v_x & v_y \end{pmatrix}^{\mathsf T}
$$

Alongside this it keeps a $4\times4$ **covariance matrix** $P$, which describes how uncertain each part of the estimate is and how the uncertainties are related.

**Step 1: predict.** At every physics step ($\Delta t = 0.25$ ms) the filter moves its estimate forward using physics. Position changes by velocity times time, and velocity changes by acceleration times time. The acceleration is the gravity-driven acceleration on the tilted plate from section A.1, $\mathbf u = k\,(R^{\mathsf T}\mathbf g)_{xy}$:

$$
\mathbf x^- = F\mathbf x + B\,\mathbf u, \qquad
F = \begin{pmatrix} I_2 & \Delta t\, I_2 \\ 0 & I_2\end{pmatrix}, \qquad
B = \begin{pmatrix} \tfrac12\Delta t^2\, I_2 \\ \Delta t\, I_2\end{pmatrix}
$$

The uncertainty grows during prediction, because the physics model is not perfect: there are kicks and pushes the filter does not know about. This is modelled as an unknown random acceleration with typical size $\sigma_a = 6\,\text{m/s}^2$, a value chosen by testing a range of values in simulation:

$$
P^- = F P F^{\mathsf T} + \sigma_a^2\, B B^{\mathsf T}
$$

After predicting, the filter makes sure the estimate stays on the plate. If the estimated position is past the edge, it is moved back to the edge and that velocity component is set to zero. This also prevents a classic trilateration mistake: two distance circles cross at *two* points, and the clamp removes the one that lies off the plate.

**Step 2: what a reading tells us.** A reading from the sensor at corner position $\mathbf s$ gives a measured distance $r$ (to the ball's *surface*) and a beam direction $\psi$ (which way the sensor was pointing). Adding the ball's radius $R_b$ converts the surface distance into the distance to the ball's centre. The filter then predicts what this reading should have been, given its current estimate of the ball's position $\mathbf p$:

$$
\mathbf z = \begin{pmatrix} r + R_b \\ \psi \end{pmatrix}, \qquad
h(\mathbf x) = \begin{pmatrix} \rho \\ \operatorname{atan2}(\Delta y,\, \Delta x)\end{pmatrix}, \qquad
\text{where } \Delta = \mathbf p - \mathbf s \text{ and } \rho = \lVert \Delta \rVert
$$

The predicted reading $h(\mathbf x)$ is simply the distance and direction from the sensor to the estimated ball position. The filter also needs to know how this predicted reading would change if the ball moved slightly. This is the **Jacobian** matrix $H$:

$$
H = \frac{\partial h}{\partial \mathbf x} = \begin{pmatrix} \Delta x/\rho & \Delta y/\rho & 0 & 0 \\ -\Delta y/\rho^2 & \Delta x/\rho^2 & 0 & 0\end{pmatrix}
$$

**How much to trust the reading.** The filter needs the uncertainty of each part of the reading.
- *The distance* is quite precise. Its uncertainty is the noise the sensor itself reports, with a minimum of 4 mm.
- *The direction* is vague. The sensor only knows the ball was *somewhere* inside its cone, and the ball's own width adds to this. If the ball is equally likely to be anywhere within an angle $\pm a$ of the beam centre, the standard deviation of its direction is $a/\sqrt3$ (a standard result for a uniform spread).

$$
\sigma_r = \max(\sigma,\; 4\,\text{mm}), \qquad
\sigma_b = \frac{\theta_{rx}(\rho) + \arcsin(R_b/\rho)}{\sqrt3}, \qquad
R_{meas} = \begin{pmatrix} \sigma_r^2 & 0 \\ 0 & \sigma_b^2 \end{pmatrix}
$$

To see what this means in practice, take a ball 20 cm from a sensor. The direction uncertainty is $\sigma_b = (10.8° + 5.7°)/\sqrt3 \approx 9.5°$. At 20 cm that is a sideways uncertainty of about 33 mm, while the distance uncertainty is only 4 to 8 mm. **A single reading therefore places the ball on a thin arc about 6 cm long, not at a point.**

**Step 3: check and correct.** The filter computes the **innovation** $\mathbf y$, the difference between the actual reading and the predicted reading, and its expected spread $S$:

$$
\mathbf y = \mathbf z - h(\mathbf x^-), \qquad S = H P^- H^{\mathsf T} + R_{meas}
$$

(The direction part of $\mathbf y$ is wrapped into the range $-\pi$ to $\pi$, so that, for example, 359° and 1° count as 2° apart.)

Before using the reading, the filter checks that it is plausible. The quantity $\mathbf y^{\mathsf T} S^{-1}\mathbf y$ measures how surprising the reading is, taking all uncertainties into account. If it exceeds 9.21, a reading this far off would happen by chance less than 1% of the time, so it is treated as an error and rejected:

$$
\text{reject the reading if } \mathbf y^{\mathsf T} S^{-1}\mathbf y > 9.21
$$

Otherwise, the filter computes the **Kalman gain** $K$, which decides how far to move the estimate towards the reading, and updates both the estimate and its uncertainty:

$$
K = P^- H^{\mathsf T} S^{-1}, \qquad \mathbf x = \mathbf x^- + K\mathbf y, \qquad P = (I - K H)\,P^-
$$

The gain is large when the filter is unsure of its own estimate and the reading is precise, and small in the opposite case.

**The extra rules, in equations.**
- *Background rejection:* when the simulation starts, each sensor records the empty-plate distance $r_{bg}$ for every position in its sweep. A reading is ignored if $r > r_{bg} - 25\,\text{mm}$, i.e. if it is not clearly closer than the empty plate.
- *Starting up:* the first accepted reading places the ball at $\mathbf p = \mathbf s + (r+R_b)(\cos\psi,\, \sin\psi)$, with zero velocity and an initial uncertainty of 5 cm in position and 0.3 m/s in velocity.
- *Losing the ball:* if no reading has been accepted for 0.3 seconds, the next reading restarts the filter.

**Why several sensors help, and where they struggle.** With two sensors, the ball's position is pinned down by the two distance readings, and the accuracy depends on the angle $\gamma$ between the two lines from the sensors to the ball. Using the theory of least-squares estimation, the position uncertainty in the worst direction is:

$$
\sigma_{worst} = \frac{\sigma_r}{\sqrt{1 - |\cos\gamma|}}
$$

When the two lines cross at a right angle ($\gamma = 90°$), the uncertainty is simply $\sigma_r$. But when the ball is close to the straight line through both sensors, the lines are nearly parallel ($\gamma$ near 0° or 180°), $|\cos\gamma|$ approaches 1, and the uncertainty grows very large. This is exactly the weak spot the experiments found (the "hold_x2" scenario, 13.5 mm). With only one sensor, only the 33 mm sideways uncertainty from the beam width remains, which matches the 29 mm average error measured with a single sensor.

### A.7 The sweep pattern (`bop_env/scan.py`)

Each sensor sweeps a sector of $\Sigma = 90°$ in steps of $\delta_s$, taking one reading (lasting $T_r = 10\,\text{ms}$) per step. The number of readings per sweep, the time for one sweep in one direction, and the time for a full there-and-back cycle are:

$$
N = \frac{\Sigma}{\delta_s}, \qquad T_{sw} = N\,T_r, \qquad \text{full cycle} = 2T_{sw}
$$

The beam moves back and forth in a **triangle wave**: steadily one way, then steadily back. Each sensor $i$ (of $n_s$ sensors) is delayed by its own offset $o_i$, so that the sensors are evenly spread out in time:

$$
\varphi = (t - o_i) \bmod 2T_{sw}, \qquad o_i = \frac{i\,T_{sw}}{n_s}
$$

$$
\psi_{loc}(t) = \begin{cases} \Sigma\,\varphi/T_{sw} & \text{for the outward half } (\varphi < T_{sw}) \\ \Sigma\,(2 - \varphi/T_{sw}) & \text{for the return half} \end{cases}
$$

Here $\varphi$ is how far the sensor is through its cycle, and $\psi_{loc}$ is its angle within its 90° sector. The beam's actual direction on the plate is $\psi = \psi_{bis} - \Sigma/2 + \psi_{loc}$, where $\psi_{bis}$ is the direction of the diagonal from that corner towards the plate centre: 45°, 135°, 225° or 315°. The scan servo is commanded to $q_{scan} = \psi_{loc} - \Sigma/2$, so that angle 0 points the beam straight along the diagonal.

With the default settings, $\delta_s = 18°$ gives $N = 5$ readings per sweep, $T_{sw} = 50$ ms, and the four sensors are offset from each other by 12.5 ms.

### A.8 How fast can a sweeping sensor find the ball? (derivations)

**Readings are not the same as position updates.** A sensor takes $f_s = 1/T_r$ readings per second (100 for the VL53L4CD). But a reading only tells us where the ball is if the beam happens to be pointing at it. A sweeping sensor that divides its attention among $N$ directions passes over the ball roughly once per sweep, so the rate at which it actually updates the ball's position is:

$$
f_{upd} \approx \frac{f_s}{N} = \frac{1}{N\,T_r} \qquad \text{(per sensor)}
$$

Turning this around, to support a controller running at $f_c$ updates per second, the sensor needs $f_s \ge f_c \times N$ readings per second.

**Case 1: one sensor spinning a full circle.** With a reading every 10°, a full turn has $N = 360/10 = 36$ directions:

$$
f_{upd} = \frac{100\ \text{Hz}}{36} \approx 2.8\ \text{Hz}, \qquad \text{while 50 Hz would need } f_s \ge 50 \times 36 = 1800\ \text{readings/s}
$$

No sensor of this type comes close to 1800 readings per second. Worse, a sensor at the edge of the plate spends about three quarters of each turn facing away from the plate entirely. This is why the design was rejected.

**Case 2: four corner sensors, each sweeping 90°, staggered in time.** Each sensor passes over the ball once per one-way sweep. Because the four sensors are spread evenly in time, their passes interleave, and the combined update rate is:

$$
f_{upd} \approx \frac{n_s}{T_{sw}} = \frac{n_s\,\delta_s}{\Sigma\,T_r}
$$

| Step size $\delta_s$ | Readings per sweep $N$ | Sweep time $T_{sw}$ | Updates per sensor | Updates, 4 sensors combined |
|---|---|---|---|---|
| 18° | 5 | 50 ms | 20 per second | **80 per second** |
| 10° | 9 | 90 ms | 11 per second | **44 per second** |

Larger steps give faster updates, but the steps must not be larger than the beam width (about 18°), or the beam would jump over gaps where it could miss the ball entirely. So **18° is the largest step that still covers the sector without gaps**, and it gives the highest update rate. This agrees with the experiments, where 18° steps tracked 1.2 mm better than 10° steps.

**Case 3: how fast must the scan servo turn?** To sweep 90° in 50 ms, the beam must turn at:

$$
\omega_{req} = \frac{\Sigma}{T_{sw}} = \frac{90°}{0.050\ \text{s}} = 1800° \text{ per second}
$$

and reverse direction instantly at each end. A hobby servo's top speed is $\omega_0 = 60°/t_{60}$:
- the SG90 ($t_{60} = 0.10$ s) reaches 600° per second;
- the MG996R at 6 V reaches 367° per second;
- the DS3218 at 6.8 V reaches 430° per second.

These are **3 to 5 times too slow**, even before accounting for the time a servo needs to speed up, slow down and reverse.

A second problem comes from the 50 Hz command rate. The servo receives a new target only every 20 ms, so a 50 ms sweep gets just $50 / 20 = 2.5$ target updates per sweep, far too few to trace a smooth triangle wave. Together, these two effects explain why the simulated SG90 sweeps only about 12° of the commanded 90° (section 6.2).

**Case 4: the best a hobby servo could do.** If we slow the sweep down until the servo can keep up, the sweep time becomes $T_{sw} \ge \Sigma/\omega_0$, and the update rate is at most:

$$
f_{upd} \le \frac{n_s\,\omega_0}{\Sigma} = \frac{4 \times 600°/\text{s}}{90°} \approx 27\ \text{Hz} \qquad (\text{SG90, as an upper bound})
$$

with a sweep of 150 ms (15 readings of 6° each). This is about half of the 50 Hz target. To reach 50 Hz with four sensors, the sweep motor would need to turn at least:

$$
\omega \ge \frac{50 \times 90°}{4} = 1125° \text{ per second}
$$

None of the hobby servos on the list reaches this. A stepper motor or a galvanometer mirror would.

**Case 5: how far the ball moves while being measured.** A ball moving at speed $v$ travels $v\,T_r$ during one reading and $v\,T_{sw}$ during one sweep. At 0.5 m/s, that is 5 mm per reading and 25 mm per sweep (50 mm for a 100 ms scan). The blurring within a single reading is small. The bigger problem is **staleness**: by the time the next reading arrives, the ball has moved on.

### A.9 The controller and the reward

**The PID controller** (`control/pid.py`) works in two stages. First, it decides what acceleration it wants the ball to have, based on the position error $\mathbf e = \hat{\mathbf p} - \mathbf p^*$ (estimated position minus target), the estimated velocity, and the accumulated error over time:

$$
\mathbf a^* = -k_p\,\mathbf e \;-\; k_d\,\hat{\mathbf v} \;-\; k_i\!\int\!\mathbf e\,dt, \qquad k_p = 6,\quad k_d = 3.5,\quad k_i = 0.5
$$

The first term pulls the ball back towards the target, the second damps its motion so it does not overshoot, and the third removes any steady drift.

Second, it works out what plate tilt would produce that acceleration. From section A.1, a roll $\phi$ and pitch $\vartheta$ give ball accelerations $a_y = -k g\sin\phi$ and $a_x = k g \sin\vartheta\cos\phi$. Solving these for the angles:

$$
\phi = -\arcsin\frac{a^*_y}{k g}, \qquad \vartheta = \arcsin\frac{a^*_x}{k g\cos\phi}
$$

Both angles are limited to ±8°. The inverse kinematics of section A.3 then turns this tilt into three servo angles $q$, which are divided by 80° to give the action in the range −1 to +1.

**The reinforcement-learning reward** (`bop_env/env.py`) is given every 20 ms. With the ball at $\mathbf p$ moving at $\mathbf v$, the target at $\mathbf p^*$, half the plate width $h_p = 0.15$ m, and the agent's action $\mathbf a$:

$$
r_t = 1 \;-\; w_{pos}\frac{\lVert \mathbf p - \mathbf p^*\rVert}{h_p} \;-\; w_{vel}\lVert\mathbf v\rVert \;-\; w_{rate}\lVert \mathbf a_t - \mathbf a_{t-1}\rVert^2 \;-\; w_{act}\lVert\mathbf a_t\rVert^2
$$

The terms are, in order: a reward for staying on the plate, a penalty for distance from the target (measured in half-plate widths), a penalty for ball speed, a penalty for jerky changes in the commands, and an optional penalty for large commands. The default weights are $w_{pos} = 1$, $w_{vel} = 0.1$, $w_{rate} = 0.05$ and $w_{act} = 0$. If the ball falls off, the reward is $-20$ and the episode ends.
