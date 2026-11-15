"""Warp kernels for the batched MuJoCo Warp ball-on-plate simulation."""
import warp as wp


@wp.func
def quat_to_mat(q: wp.vec4f) -> wp.mat33f:
    w, x, y, z = q[0], q[1], q[2], q[3]
    return wp.mat33f(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w),
        2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
        2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y))


@wp.func
def rz(a: float) -> wp.mat33f:
    c, s = wp.cos(a), wp.sin(a)
    return wp.mat33f(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)


@wp.func
def ry(a: float) -> wp.mat33f:
    c, s = wp.cos(a), wp.sin(a)
    return wp.mat33f(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c)


@wp.func
def wrap_angle(a: float) -> float:
    return (a + wp.pi) - wp.floor((a + wp.pi) / (2.0 * wp.pi)) * (2.0 * wp.pi) - wp.pi


@wp.func
def local_yaw_deg(t: float, i: int, sector_deg: float, step_deg: float,
                  reading_time: float, sweep_time: float, phase_offset: float,
                  step_mode: int) -> float:
    period = 2.0 * sweep_time
    ph = t - phase_offset
    ph = ph - wp.floor(ph / period) * period
    n = int(wp.round(sector_deg / step_deg))
    if step_mode != 0:
        k = int(ph / reading_time)
        if k >= 2 * n:
            k = 2 * n - 1
        if k < n:
            return (float(k) + 0.5) * step_deg
        return (float(2 * n - k) - 0.5) * step_deg
    if ph < sweep_time:
        return sector_deg * ph / sweep_time
    return sector_deg * (2.0 - ph / sweep_time)


@wp.func
def commanded_yaw(t: float, i: int, sector_deg: float, step_deg: float,
                  reading_time: float, sweep_time: float, phase_offset: float,
                  step_mode: int, bisector_deg: float) -> float:
    local = local_yaw_deg(t, i, sector_deg, step_deg, reading_time,
                          sweep_time, phase_offset, step_mode)
    return wp.radians(bisector_deg - 0.5 * sector_deg + local)


@wp.func
def scan_servo_angle(t: float, i: int, sector_deg: float, step_deg: float,
                     reading_time: float, sweep_time: float, phase_offset: float,
                     step_mode: int) -> float:
    local = local_yaw_deg(t, i, sector_deg, step_deg, reading_time,
                          sweep_time, phase_offset, step_mode)
    return wp.radians(local - 0.5 * sector_deg)


@wp.func
def rx_half(dist: float, near: float, far: float,
            near_dist: float, far_dist: float) -> float:
    x = (dist - near_dist) / (far_dist - near_dist)
    x = wp.clamp(x, 0.0, 1.0)
    return near + x * (far - near)


@wp.func
def emission_weight(theta: float, foi_half: float) -> float:
    w = foi_half
    return wp.exp(-2.0 * theta * theta / (w * w)) / (wp.pi * w * w / 2.0)


@wp.func
def smooth_at(hist: wp.array3d(dtype=wp.float32), w: int, s: int, k: int,
              weight: wp.array(dtype=wp.float32), taps: int,
              nbins: int) -> float:
    val = float(0.0)
    mid = taps // 2
    for j in range(taps):
        idx = k + j - mid
        if idx >= 0 and idx < nbins:
            val += hist[w, s, idx] * weight[j]
    return val


@wp.kernel
def sensor_poses(qpos: wp.array2d(dtype=wp.float32), sim_time: wp.array(dtype=wp.float32),
                 plate_q: int, corner_x: wp.array(dtype=wp.float32),
                 corner_y: wp.array(dtype=wp.float32), bisector: wp.array(dtype=wp.float32),
                 scan_q: wp.array(dtype=wp.int32), scan_lash: wp.array(dtype=wp.int32),
                 sector_deg: float, step_deg: float, reading_time: float,
                 sweep_time: float, phase_scale: float, step_mode: int,
                 scan_servo: int, yaw_measured: int, feedback_noise: float,
                 mount_height: float, pitch: float, seed: int,
                 out_origin: wp.array2d(dtype=wp.vec3f),
                 out_rot: wp.array2d(dtype=wp.mat33f),
                 out_beam: wp.array2d(dtype=wp.float32),
                 out_bearing: wp.array2d(dtype=wp.float32)):
    w = wp.tid()
    q = wp.vec4f(qpos[w, plate_q + 3], qpos[w, plate_q + 4],
                 qpos[w, plate_q + 5], qpos[w, plate_q + 6])
    R = quat_to_mat(q)
    t = sim_time[w]
    ns = out_origin.shape[1]
    for i in range(ns):
        phase = float(i) * sweep_time * phase_scale
        cmd = commanded_yaw(t, i, sector_deg, step_deg, reading_time,
                            sweep_time, phase, step_mode, bisector[i])
        if scan_servo != 0:
            horn = qpos[w, scan_q[i]] + qpos[w, scan_lash[i]]
            beam = wp.radians(bisector[i]) + horn
        else:
            beam = cmd
        Rs = R @ rz(beam) @ ry(pitch)
        p = wp.vec3f(qpos[w, plate_q], qpos[w, plate_q + 1], qpos[w, plate_q + 2])
        origin = p + R @ wp.vec3f(corner_x[i], corner_y[i], mount_height)
        out_origin[w, i] = origin
        out_rot[w, i] = Rs
        out_beam[w, i] = beam
        if yaw_measured == 0 or scan_servo == 0:
            out_bearing[w, i] = cmd
        else:
            state = wp.rand_init(seed, w * 100003 + i * 997 + int(t * 10000.0))
            out_bearing[w, i] = beam + wp.randn(state) * feedback_noise


@wp.kernel
def primary_rays(origin: wp.array2d(dtype=wp.vec3f),
                 rot: wp.array2d(dtype=wp.mat33f),
                 local: wp.array(dtype=wp.vec3f), nray: int,
                 pnt: wp.array2d(dtype=wp.vec3f),
                 vec: wp.array2d(dtype=wp.vec3f)):
    w, idx = wp.tid()
    s = idx // nray
    k = idx % nray
    pnt[w, idx] = origin[w, s]
    vec[w, idx] = rot[w, s] @ local[k]


@wp.kernel
def aux_rays(origin: wp.array2d(dtype=wp.vec3f),
             ppnt: wp.array2d(dtype=wp.vec3f),
             pvec: wp.array2d(dtype=wp.vec3f),
             pdist: wp.array2d(dtype=wp.float32),
             nray: int, ns: int,
             apnt: wp.array2d(dtype=wp.vec3f),
             avec: wp.array2d(dtype=wp.vec3f),
             valid: wp.array2d(dtype=wp.int32)):
    w, idx = wp.tid()
    per = (ns - 1) * (nray + 1)
    r = idx // per
    rem = idx % per
    jj = rem // (nray + 1)
    k = rem % (nray + 1)
    j = jj
    if jj >= r:
        j = jj + 1
    rx = origin[w, r]
    ok = 0
    d = wp.vec3f(0.0, 0.0, 1.0)
    if k < nray:
        d0 = pdist[w, j * nray + k]
        if d0 > 0.0:
            j0 = j * nray + k
            q = ppnt[w, j0] + pvec[w, j0] * d0
            v = q - rx
            l = wp.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
            if l > 1e-8:
                d = v / l
                ok = 1
    else:
        v = origin[w, j] - rx
        l = wp.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        if l > 1e-8:
            d = v / l
            ok = 1
    apnt[w, idx] = rx
    avec[w, idx] = d
    valid[w, idx] = ok


@wp.kernel
def leg_servo_step(qpos: wp.array2d(dtype=wp.float32), qvel: wp.array2d(dtype=wp.float32),
                   ctrl: wp.array2d(dtype=wp.float32), sim_time: wp.array(dtype=wp.float32),
                   qidx: wp.array(dtype=wp.int32), lidx: wp.array(dtype=wp.int32),
                   didx: wp.array(dtype=wp.int32), sent: wp.array2d(dtype=wp.float32),
                   latched: wp.array2d(dtype=wp.float32),
                   next_frame: wp.array2d(dtype=wp.float32),
                   saturated: wp.array2d(dtype=wp.int32),
                   frame_dt: float, us_per_rad: float, centre: float,
                   pmin: float, pmax: float, res: float, prop_band_us: float,
                   deadband_us: float, w_noload: float, t_stall: float,
                   host_speed: float):
    w = wp.tid()
    t = sim_time[w]
    for i in range(ctrl.shape[1]):
        if t >= next_frame[w, i] - 1e-9:
            nxt = next_frame[w, i] + frame_dt
            if nxt < t:
                nxt = t
            next_frame[w, i] = nxt
            new = sent[w, i]
            if host_speed > 0.0:
                step = host_speed * frame_dt * us_per_rad
                new = wp.clamp(new, latched[w, i] - step, latched[w, i] + step)
            latched[w, i] = new
        horn = qpos[w, qidx[i]] + qpos[w, lidx[i]]
        err = latched[w, i] - (centre + horn * us_per_rad)
        duty = wp.clamp(err / prop_band_us, -1.0, 1.0)
        if wp.abs(err) <= deadband_us * 0.5:
            duty = 0.0
        saturated[w, i] = 1 if wp.abs(duty) >= 1.0 else 0
        ctrl[w, i] = t_stall * (duty - qvel[w, didx[i]] / w_noload)


@wp.kernel
def scan_servo_step(qpos: wp.array2d(dtype=wp.float32), qvel: wp.array2d(dtype=wp.float32),
                    ctrl: wp.array2d(dtype=wp.float32), sim_time: wp.array(dtype=wp.float32),
                    qidx: wp.array(dtype=wp.int32), lidx: wp.array(dtype=wp.int32),
                    didx: wp.array(dtype=wp.int32), ctrl_offset: int,
                    latched: wp.array2d(dtype=wp.float32),
                    next_frame: wp.array2d(dtype=wp.float32),
                    saturated: wp.array2d(dtype=wp.int32),
                    sector_deg: float, step_deg: float, reading_time: float,
                    sweep_time: float, phase_scale: float, step_mode: int,
                    frame_dt: float, us_per_rad: float, centre: float,
                    pmin: float, pmax: float, res: float, prop_band_us: float,
                    deadband_us: float, w_noload: float, t_stall: float):
    w = wp.tid()
    t = sim_time[w]
    for i in range(latched.shape[1]):
        q = scan_servo_angle(t, i, sector_deg, step_deg, reading_time,
                             sweep_time, float(i) * sweep_time * phase_scale, step_mode)
        p = centre + q * us_per_rad
        p = wp.round(p / res) * res
        p = wp.clamp(p, pmin, pmax)
        if t >= next_frame[w, i] - 1e-9:
            nxt = next_frame[w, i] + frame_dt
            if nxt < t:
                nxt = t
            next_frame[w, i] = nxt
            latched[w, i] = p
        horn = qpos[w, qidx[i]] + qpos[w, lidx[i]]
        err = latched[w, i] - (centre + horn * us_per_rad)
        duty = wp.clamp(err / prop_band_us, -1.0, 1.0)
        if wp.abs(err) <= deadband_us * 0.5:
            duty = 0.0
        saturated[w, i] = 1 if wp.abs(duty) >= 1.0 else 0
        ctrl[w, ctrl_offset + i] = t_stall * (duty - qvel[w, didx[i]] / w_noload)


@wp.kernel
def force_step(qpos: wp.array2d(dtype=wp.float32), xfrc: wp.array2d(dtype=wp.spatial_vectorf),
               sim_time: wp.array(dtype=wp.float32), plate_q: int, ball_body: int,
               push: wp.array2d(dtype=wp.float32), kicks: wp.array3d(dtype=wp.float32),
               nkicks: wp.array(dtype=wp.int32), dt: float, seed: int,
               push_sigma: float, push_tau: float):
    w = wp.tid()
    t = sim_time[w]
    if push_sigma > 0.0:
        state = wp.rand_init(seed, w * 200003 + int(t / dt))
        n0 = wp.randn(state)
        n1 = wp.randn(state)
        push[w, 0] = push[w, 0] - push[w, 0] / push_tau * dt + push_sigma * wp.sqrt(2.0 * dt / push_tau) * n0
        push[w, 1] = push[w, 1] - push[w, 1] / push_tau * dt + push_sigma * wp.sqrt(2.0 * dt / push_tau) * n1
    fx = float(push[w, 0])
    fy = float(push[w, 1])
    for k in range(nkicks[w]):
        if kicks[w, k, 0] <= t and t < kicks[w, k, 1]:
            fx += kicks[w, k, 2]
            fy += kicks[w, k, 3]
    q = wp.vec4f(qpos[w, plate_q + 3], qpos[w, plate_q + 4],
                 qpos[w, plate_q + 5], qpos[w, plate_q + 6])
    f = quat_to_mat(q) @ wp.vec3f(fx, fy, 0.0)
    xfrc[w, ball_body] = wp.spatial_vectorf(f[0], f[1], f[2], 0.0, 0.0, 0.0)


@wp.kernel
def truth_step(qpos: wp.array2d(dtype=wp.float32), qvel: wp.array2d(dtype=wp.float32),
               sim_time: wp.array(dtype=wp.float32), plate_q: int, ball_q: int,
               ball_dof: int, half: float, escaped: wp.array(dtype=wp.int32),
               escape_t: wp.array(dtype=wp.float32),
               rel_pos: wp.array2d(dtype=wp.float32),
               rel_vel: wp.array2d(dtype=wp.float32)):
    w = wp.tid()
    q = wp.vec4f(qpos[w, plate_q + 3], qpos[w, plate_q + 4],
                 qpos[w, plate_q + 5], qpos[w, plate_q + 6])
    R = quat_to_mat(q)
    p = wp.vec3f(qpos[w, ball_q] - qpos[w, plate_q],
                 qpos[w, ball_q + 1] - qpos[w, plate_q + 1],
                 qpos[w, ball_q + 2] - qpos[w, plate_q + 2])
    rp = wp.transpose(R) @ p
    rv = wp.transpose(R) @ wp.vec3f(qvel[w, ball_dof], qvel[w, ball_dof + 1], qvel[w, ball_dof + 2])
    rel_pos[w, 0] = rp[0]
    rel_pos[w, 1] = rp[1]
    rel_pos[w, 2] = rp[2]
    rel_vel[w, 0] = rv[0]
    rel_vel[w, 1] = rv[1]
    rel_vel[w, 2] = rv[2]
    if escaped[w] == 0 and (wp.abs(rp[0]) > half or wp.abs(rp[1]) > half or rp[2] < 0.0):
        escaped[w] = 1
        escape_t[w] = sim_time[w]


@wp.kernel
def tracker_predict(qpos: wp.array2d(dtype=wp.float32), plate_q: int,
                    x: wp.array2d(dtype=wp.float32), P: wp.array2d(dtype=wp.float32),
                    dt: float, roll_factor: float, sigma: float, limit: float):
    w = wp.tid()
    q = wp.vec4f(qpos[w, plate_q + 3], qpos[w, plate_q + 4],
                 qpos[w, plate_q + 5], qpos[w, plate_q + 6])
    R = quat_to_mat(q)
    g = wp.transpose(R) @ wp.vec3f(0.0, 0.0, -9.81)
    ax = roll_factor * g[0]
    ay = roll_factor * g[1]
    F = wp.mat44f(1.0, 0.0, dt, 0.0,
                  0.0, 1.0, 0.0, dt,
                  0.0, 0.0, 1.0, 0.0,
                  0.0, 0.0, 0.0, 1.0)
    Pm = wp.mat44f(P[w, 0], P[w, 1], P[w, 2], P[w, 3],
                   P[w, 4], P[w, 5], P[w, 6], P[w, 7],
                   P[w, 8], P[w, 9], P[w, 10], P[w, 11],
                   P[w, 12], P[w, 13], P[w, 14], P[w, 15])
    q00 = dt * dt * dt * dt / 4.0 * sigma * sigma
    q02 = dt * dt * dt / 2.0 * sigma * sigma
    q11 = q00
    q13 = q02
    q20 = q02
    q22 = dt * dt * sigma * sigma
    q31 = q13
    q33 = q22
    Q = wp.mat44f(q00, 0.0, q02, 0.0,
                  0.0, q11, 0.0, q13,
                  q20, 0.0, q22, 0.0,
                  0.0, q31, 0.0, q33)
    Pn = F @ Pm @ wp.transpose(F) + Q
    xv = wp.vec4f(x[w, 0] + x[w, 2] * dt + 0.5 * ax * dt * dt,
                  x[w, 1] + x[w, 3] * dt + 0.5 * ay * dt * dt,
                  x[w, 2] + ax * dt, x[w, 3] + ay * dt)
    for r in range(2):
        if wp.abs(xv[r]) > limit:
            xv[r] = wp.sign(xv[r]) * limit
            xv[r + 2] = 0.0
    for r in range(4):
        x[w, r] = xv[r]
        for c in range(4):
            P[w, r * 4 + c] = Pn[r, c]


@wp.func
def tracker_update(w: int, s: int, valid: int, rng_range: float, sigma: float,
                   yaw: float, t: float, slot: int,
                   corner_x: wp.array(dtype=wp.float32), corner_y: wp.array(dtype=wp.float32),
                   bg: wp.array2d(dtype=wp.float32), x: wp.array2d(dtype=wp.float32),
                   P: wp.array2d(dtype=wp.float32), ok: wp.array(dtype=wp.int32),
                   tlast: wp.array(dtype=wp.float32),
                   R_ball: float, limit: float, lost_after: float,
                   range_floor: float, bg_margin: float, gate: float,
                   init_pos: float, init_vel: float, sensor_near: float,
                   sensor_far: float, sensor_near_dist: float, sensor_far_dist: float):
    if valid == 0:
        return
    b = bg[s, slot]
    if b == b and rng_range > b - bg_margin:
        return
    if ok[w] != 0 and t - tlast[w] > lost_after:
        ok[w] = 0
    cx = corner_x[s]
    cy = corner_y[s]
    if ok[w] == 0:
        x[w, 0] = cx + rng_range * wp.cos(yaw)
        x[w, 1] = cy + rng_range * wp.sin(yaw)
        x[w, 2] = 0.0
        x[w, 3] = 0.0
        for r in range(4):
            for c in range(4):
                P[w, r * 4 + c] = 0.0
        P[w, 0] = init_pos * init_pos
        P[w, 5] = init_pos * init_pos
        P[w, 10] = init_vel * init_vel
        P[w, 15] = init_vel * init_vel
        ok[w] = 1
        tlast[w] = t
        return
    dx0 = x[w, 0] - cx
    dx1 = x[w, 1] - cy
    rho = wp.sqrt(dx0 * dx0 + dx1 * dx1)
    if rho < R_ball * 1.01:
        rho = R_ball * 1.01
    h00 = dx0 / rho
    h01 = dx1 / rho
    h10 = -dx1 / (rho * rho)
    h11 = dx0 / (rho * rho)
    p00, p01, p02, p03 = P[w, 0], P[w, 1], P[w, 2], P[w, 3]
    p10, p11, p12, p13 = P[w, 4], P[w, 5], P[w, 6], P[w, 7]
    p20, p21, p22, p23 = P[w, 8], P[w, 9], P[w, 10], P[w, 11]
    p30, p31, p32, p33 = P[w, 12], P[w, 13], P[w, 14], P[w, 15]
    a00 = p00 * h00 + p01 * h10
    a01 = p00 * h01 + p01 * h11
    a10 = p10 * h00 + p11 * h10
    a11 = p10 * h01 + p11 * h11
    a20 = p20 * h00 + p21 * h10
    a21 = p20 * h01 + p21 * h11
    a30 = p30 * h00 + p31 * h10
    a31 = p30 * h01 + p31 * h11
    sr = wp.max(sigma, range_floor)
    rb = rx_half(rho, sensor_near, sensor_far, sensor_near_dist, sensor_far_dist)
    sb = (rb + wp.asin(R_ball / rho)) / wp.sqrt(3.0)
    s00 = h00 * a00 + h01 * a10 + sr * sr
    s01 = h00 * a01 + h01 * a11
    s10 = h10 * a00 + h11 * a10
    s11 = h10 * a01 + h11 * a11 + sb * sb
    det = s00 * s11 - s01 * s10
    if wp.abs(det) < 1e-18:
        return
    i00, i01 = s11 / det, -s01 / det
    i10, i11 = -s10 / det, s00 / det
    yr = rng_range + R_ball - rho
    yb = wrap_angle(yaw - wp.atan2(dx1, dx0))
    gate_v = yr * (i00 * yr + i01 * yb) + yb * (i10 * yr + i11 * yb)
    if gate_v > gate:
        return
    k00 = a00 * i00 + a01 * i10
    k01 = a00 * i01 + a01 * i11
    k10 = a10 * i00 + a11 * i10
    k11 = a10 * i01 + a11 * i11
    k20 = a20 * i00 + a21 * i10
    k21 = a20 * i01 + a21 * i11
    k30 = a30 * i00 + a31 * i10
    k31 = a30 * i01 + a31 * i11
    x[w, 0] += k00 * yr + k01 * yb
    x[w, 1] += k10 * yr + k11 * yb
    x[w, 2] += k20 * yr + k21 * yb
    x[w, 3] += k30 * yr + k31 * yb
    kh00 = k00 * h00 + k01 * h10
    kh01 = k00 * h01 + k01 * h11
    kh02 = 0.0
    kh03 = 0.0
    kh10 = k10 * h00 + k11 * h10
    kh11 = k10 * h01 + k11 * h11
    kh12 = 0.0
    kh13 = 0.0
    kh20 = k20 * h00 + k21 * h10
    kh21 = k20 * h01 + k21 * h11
    kh22 = 0.0
    kh23 = 0.0
    kh30 = k30 * h00 + k31 * h10
    kh31 = k30 * h01 + k31 * h11
    kh32 = 0.0
    kh33 = 0.0
    n00 = p00 - (kh00 * p00 + kh01 * p10)
    n01 = p01 - (kh00 * p01 + kh01 * p11)
    n02 = p02 - (kh00 * p02 + kh01 * p12)
    n03 = p03 - (kh00 * p03 + kh01 * p13)
    n10 = p10 - (kh10 * p00 + kh11 * p10)
    n11 = p11 - (kh10 * p01 + kh11 * p11)
    n12 = p12 - (kh10 * p02 + kh11 * p12)
    n13 = p13 - (kh10 * p03 + kh11 * p13)
    n20 = p20 - (kh20 * p00 + kh21 * p10)
    n21 = p21 - (kh20 * p01 + kh21 * p11)
    n22 = p22 - (kh20 * p02 + kh21 * p12)
    n23 = p23 - (kh20 * p03 + kh21 * p13)
    n30 = p30 - (kh30 * p00 + kh31 * p10)
    n31 = p31 - (kh30 * p01 + kh31 * p11)
    n32 = p32 - (kh30 * p02 + kh31 * p12)
    n33 = p33 - (kh30 * p03 + kh31 * p13)
    P[w, 0] = n00
    P[w, 1] = n01
    P[w, 2] = n02
    P[w, 3] = n03
    P[w, 4] = n10
    P[w, 5] = n11
    P[w, 6] = n12
    P[w, 7] = n13
    P[w, 8] = n20
    P[w, 9] = n21
    P[w, 10] = n22
    P[w, 11] = n23
    P[w, 12] = n30
    P[w, 13] = n31
    P[w, 14] = n32
    P[w, 15] = n33
    for r in range(2):
        if wp.abs(x[w, r]) > limit:
            x[w, r] = wp.sign(x[w, r]) * limit
            x[w, r + 2] = 0.0
    tlast[w] = t
