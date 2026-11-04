"""Hobby PWM servo model. One ServoBank drives several servos of one model.

Host side: the controller sends an angle (rad, 0 = centre pulse). It becomes a
pulse width, quantised to the PWM generator step and clipped to the pulse range.
The servo latches a new pulse only once per PWM frame. An optional host speed
limit moves the latched pulse at most host_speed rad/s.

Servo side: the pot on the output shaft (horn, after the gear backlash) closes
a P loop with a deadband. The drive goes to a DC motor through the gearbox:
  duty   = clip(error / prop_band, -1, 1), 0 inside the deadband
  torque = T_stall * (duty - w / w_noload)    linear torque-speed curve
so the horn tops out at the rated no-load speed and stalls at the rated torque.
Gear backlash is a free hinge with end stops in the MJCF (see legs.py).
"""
import numpy as np

KGCM = 0.0980665  # N m per kg cm


def spec(lib, name, volts):
    """Spec of a model from servos.toml, defaults filled in, speed and torque
    interpolated at the supply voltage (clamped to the listed range)."""
    s = {**lib["default"], **lib[name], "name": name}
    s["w_noload"] = (np.pi / 3) / float(np.interp(volts, s["volts"], s["sec_per_60"]))
    s["t_stall"] = float(np.interp(volts, s["volts"], s["torque_kgcm"])) * KGCM
    s["us_per_rad"] = (s["pulse_max_us"] - s["pulse_min_us"]) / np.radians(s["travel_deg"])
    s["armature"] = s["mech_time_const_s"] * s["t_stall"] / s["w_noload"]  # reflected inertia
    s["friction"] = s["friction_frac"] * s["t_stall"]
    return s


class ServoBank:
    def __init__(self, s, n, frame_hz, res_us, host_speed=0.0):
        self.s, self.n = s, n
        self.frame_dt = 1.0 / min(frame_hz, s["max_frame_hz"])
        self.res, self.host_speed = res_us, host_speed
        self.reset(np.zeros(n))

    def reset(self, q0, t0=0.0):
        self.sent = self.to_us(q0)
        self.latched = self.sent.copy()
        self.next = t0
        self.saturated = np.zeros(self.n, bool)

    def to_us(self, q):
        s = self.s
        p = s["pulse_centre_us"] + np.asarray(q, float) * s["us_per_rad"]
        p = np.round(p / self.res) * self.res
        return np.clip(p, s["pulse_min_us"], s["pulse_max_us"])

    def to_rad(self, us):
        return (np.asarray(us) - self.s["pulse_centre_us"]) / self.s["us_per_rad"]

    def send(self, q):
        """Host sets the angle to send. Takes effect at the next PWM frame."""
        self.sent = self.to_us(q)

    def torque(self, t, q_horn, w_motor):
        """Motor torque (N m, gearbox output) at time t."""
        s = self.s
        if t >= self.next - 1e-9:  # new PWM frame
            self.next = max(self.next + self.frame_dt, t)
            new = self.sent
            if self.host_speed > 0:
                step = self.host_speed * self.frame_dt * s["us_per_rad"]
                new = np.clip(new, self.latched - step, self.latched + step)
            self.latched = new
        err = self.latched - (s["pulse_centre_us"] + q_horn * s["us_per_rad"])  # in us
        duty = np.clip(err / (np.radians(s["prop_band_deg"]) * s["us_per_rad"]), -1, 1)
        duty[np.abs(err) <= s["deadband_us"] / 2] = 0.0
        self.saturated = np.abs(duty) >= 1.0
        return s["t_stall"] * (duty - w_motor / s["w_noload"])

    def target(self):
        """Angle of the latched pulse (rad)."""
        return self.to_rad(self.latched)

    def feedback(self, q_horn, rng):
        """Pot reading (rad) if the model has a feedback wire, else NaN."""
        if self.s["feedback"] == "none":
            return np.full(self.n, np.nan)
        return q_horn + rng.normal(0, np.radians(self.s["feedback_noise_deg"]), self.n)
