"""Scan schedule: each corner sensor oscillates over its 90 deg quadrant."""
import numpy as np

from .model import BISECTOR_DEG


def reading_time(sensor):
    return sensor["timing_budget_ms"] / 1000


def sweep_time(scan, sensor):
    """One-way sweep: one reading per step."""
    return scan["sector_deg"] / scan["step_deg"] * reading_time(sensor)


def offset(i, scan, sensor):
    """Phase offset of sensor i: fitted sensors spread evenly over one sweep."""
    n = len(scan["corners"])
    return i * sweep_time(scan, sensor) / n if scan["phase_stagger"] else 0.0


def local_yaw_deg(t, i, scan, sensor):
    """Angle inside the quadrant, 0..sector, triangle wave in time."""
    T = sweep_time(scan, sensor)
    ph = (t - offset(i, scan, sensor)) % (2 * T)
    sector, step = scan["sector_deg"], scan["step_deg"]
    if scan["mode"] == "step":
        n = int(round(sector / step))
        k = int(ph // reading_time(sensor))
        return (k + 0.5) * step if k < n else (2 * n - k - 0.5) * step
    return sector * ph / T if ph < T else sector * (2 - ph / T)


def yaw(t, i, scan, sensor):
    """Plate-frame beam yaw of sensor i in rad."""
    corner = scan["corners"][i]
    deg = BISECTOR_DEG[corner] - scan["sector_deg"] / 2 + local_yaw_deg(t, i, scan, sensor)
    return np.radians(deg)


def window(t, i, scan, sensor):
    """Index of the reading window that time t falls in."""
    return int(np.floor((t - offset(i, scan, sensor)) / reading_time(sensor)))


def slot(win, scan):
    """Position of a reading inside the repeating out-and-back cycle."""
    n = int(round(scan["sector_deg"] / scan["step_deg"]))
    return win % (2 * n)


def servo_angle(t, i, scan, sensor):
    """Scan servo angle (rad) for sensor i: 0 = beam on the corner bisector."""
    return np.radians(local_yaw_deg(t, i, scan, sensor) - scan["sector_deg"] / 2)
