"""ToF history, peak selection and tracker-update kernels."""
import warp as wp

from .kernels import rx_half, emission_weight, smooth_at, tracker_update


@wp.kernel
def tof_tick(sim_time: wp.array(dtype=wp.float32),
             origin: wp.array2d(dtype=wp.vec3f),
             rot: wp.array2d(dtype=wp.mat33f),
             beam_yaw: wp.array2d(dtype=wp.float32),
             bearing: wp.array2d(dtype=wp.float32),
             pnt: wp.array2d(dtype=wp.vec3f),
             vec: wp.array2d(dtype=wp.vec3f),
             dist: wp.array2d(dtype=wp.float32),
             geomid: wp.array2d(dtype=wp.int32),
             normal: wp.array2d(dtype=wp.vec3f),
             apnt: wp.array2d(dtype=wp.vec3f),
             avec: wp.array2d(dtype=wp.vec3f),
             adist: wp.array2d(dtype=wp.float32),
             ageom: wp.array2d(dtype=wp.int32),
             anorm: wp.array2d(dtype=wp.vec3f),
             aux_valid: wp.array2d(dtype=wp.int32),
             albedo: wp.array(dtype=wp.float32),
             housing: wp.array(dtype=wp.int32),
             ray_power: wp.array(dtype=wp.float32),
             ray_theta: wp.array(dtype=wp.float32),
             pulse: wp.array(dtype=wp.float32),
             hist: wp.array3d(dtype=wp.float32),
             amb: wp.array2d(dtype=wp.float32),
             acc_yaw: wp.array2d(dtype=wp.float32),
             acc_beam: wp.array2d(dtype=wp.float32),
             acc_n: wp.array2d(dtype=wp.int32),
             wins: wp.array2d(dtype=wp.int32),
             rd_valid: wp.array2d(dtype=wp.int32),
             rd_range: wp.array2d(dtype=wp.float32),
             rd_sigma: wp.array2d(dtype=wp.float32),
             rd_signal: wp.array2d(dtype=wp.float32),
             rd_ambient: wp.array2d(dtype=wp.float32),
             rd_yaw: wp.array2d(dtype=wp.float32),
             rd_beam: wp.array2d(dtype=wp.float32),
             rd_slot: wp.array2d(dtype=wp.int32),
             rd_time: wp.array2d(dtype=wp.float32),
             rd_sensor: wp.array2d(dtype=wp.int32),
             finished: wp.array2d(dtype=wp.int32),
             read_count: wp.array(dtype=wp.int32),
             read_valid: wp.array2d(dtype=wp.int32),
             read_sensor: wp.array2d(dtype=wp.int32),
             read_range: wp.array2d(dtype=wp.float32),
             read_sigma: wp.array2d(dtype=wp.float32),
             read_signal: wp.array2d(dtype=wp.float32),
             read_ambient: wp.array2d(dtype=wp.float32),
             read_yaw: wp.array2d(dtype=wp.float32),
             read_beam: wp.array2d(dtype=wp.float32),
             read_slot: wp.array2d(dtype=wp.int32),
             read_time: wp.array2d(dtype=wp.float32),
             sensor_offset: wp.array2d(dtype=wp.float32),
             nray: int, ns: int, nbins: int, nslots: int,
             bin_m: float, max_range: float, s_ref: float,
             reading_time: float, sector_deg: float, step_deg: float,
             sweep_time: float, phase_scale: float, step_mode: int,
             offset_sigma: float, sigma_ref: float, timing_budget: float,
             ref_budget: float, sigma_threshold: float, ambient_frac: float,
             ambient_span: float, nearest_frac: float, peak_nearest: int,
             pulse_sigma: float, pulse_taps: int, interference: int,
             sensor_near: float, sensor_far: float,
             sensor_near_dist: float, sensor_far_dist: float,
             foi_half: float, ep_seed: wp.array(dtype=wp.int32)):
    w = wp.tid()
    t = sim_time[w]
    for i in range(ns):
        phase = float(i) * sweep_time * phase_scale
        win = int(wp.floor((t - phase) / reading_time))
        old = wins[w, i]
        finished[w, i] = 0

        if old == -999999:
            wins[w, i] = win
        if old != -999999 and win != old:
            n = acc_n[w, i]
            if n > 0:
                slot = old % nslots
                if slot < 0:
                    slot += nslots
                hsum = float(0.0)
                for b in range(nbins):
                    hsum += hist[w, i, b]
                signal = float(0.0)
                peak = int(0)
                if hsum > 0.0:
                    top = float(-1.0)
                    for b in range(nbins):
                        sm = smooth_at(hist, w, i, b, pulse, pulse_taps, nbins) / float(n)
                        if sm > top:
                            top = sm
                    if peak_nearest != 0:
                        for b in range(1, nbins - 1):
                            sm = smooth_at(hist, w, i, b, pulse, pulse_taps, nbins) / float(n)
                            sl = smooth_at(hist, w, i, b - 1, pulse, pulse_taps, nbins) / float(n)
                            sr = smooth_at(hist, w, i, b + 1, pulse, pulse_taps, nbins) / float(n)
                            if sm >= nearest_frac * top and sm >= sl and sm >= sr:
                                peak = b
                                break
                    else:
                        best = float(-1.0)
                        for b in range(nbins):
                            sm = smooth_at(hist, w, i, b, pulse, pulse_taps, nbins) / float(n)
                            if sm > best:
                                best = sm
                                peak = b
                    ww = int(2.0 * pulse_sigma / bin_m)
                    lo = max(peak - ww, 0)
                    hi = min(peak + ww + 1, nbins)
                    for b in range(lo, hi):
                        signal += hist[w, i, b] / float(n)
                av = amb[w, i] / float(n) + ambient_frac * s_ref
                valid = int(0)
                rng_range = float(0.0)
                sig = float(0.0)
                if signal > 0.0:
                    mean_range = float(0.0)
                    lo = max(peak - int(2.0 * pulse_sigma / bin_m), 0)
                    hi = min(peak + int(2.0 * pulse_sigma / bin_m) + 1, nbins)
                    for b in range(lo, hi):
                        mean_range += (hist[w, i, b] / float(n)) * (float(b) + 0.5) * bin_m
                    mean_range = mean_range / signal
                    amb_peak = av * float(hi - lo) * bin_m / ambient_span
                    sig = sigma_ref * wp.sqrt(s_ref / signal)
                    sig *= wp.sqrt(ref_budget / timing_budget)
                    sig *= wp.sqrt(1.0 + amb_peak / signal)
                    if sig <= sigma_threshold:
                        valid = 1
                        state = wp.rand_init(ep_seed[w], i * 10007 + old)
                        rng_range = mean_range + sensor_offset[w, i] + wp.randn(state) * sig
                rd_valid[w, i] = valid
                rd_range[w, i] = rng_range
                rd_sigma[w, i] = sig
                rd_signal[w, i] = signal
                rd_ambient[w, i] = av
                rd_yaw[w, i] = acc_yaw[w, i] / float(n)
                rd_beam[w, i] = acc_beam[w, i] / float(n)
                rd_slot[w, i] = slot
                rd_time[w, i] = t
                rd_sensor[w, i] = i
                finished[w, i] = 1
                k = read_count[w]
                if k < read_valid.shape[1]:
                    read_valid[w, k] = valid
                    read_sensor[w, k] = i
                    read_range[w, k] = rng_range
                    read_sigma[w, k] = sig
                    read_signal[w, k] = signal
                    read_ambient[w, k] = av
                    read_yaw[w, k] = acc_yaw[w, i] / float(n)
                    read_beam[w, k] = acc_beam[w, i] / float(n)
                    read_slot[w, k] = slot
                    read_time[w, k] = t
                    read_count[w] = k + 1
            for b in range(nbins):
                hist[w, i, b] = 0.0
            amb[w, i] = 0.0
            acc_yaw[w, i] = 0.0
            acc_beam[w, i] = 0.0
            acc_n[w, i] = 0
            wins[w, i] = win

        axis = rot[w, i] @ wp.vec3f(1.0, 0.0, 0.0)
        rx = origin[w, i]
        for k in range(nray):
            idx = i * nray + k
            d = dist[w, idx]
            if d > 0.0:
                g = geomid[w, idx]
                if g >= 0:
                    if ray_theta[k] <= rx_half(d, sensor_near, sensor_far,
                                              sensor_near_dist, sensor_far_dist):
                        cos_in = -(normal[w, idx][0] * vec[w, idx][0] +
                                   normal[w, idx][1] * vec[w, idx][1] +
                                   normal[w, idx][2] * vec[w, idx][2])
                        cos_in = wp.max(cos_in, 0.0)
                        val = ray_power[k] * albedo[g] * cos_in / (wp.pi * d * d)
                        b = int(d / bin_m)
                        if b >= 0 and b < nbins:
                            hist[w, i, b] += val
        if interference != 0:
            for jj in range(ns - 1):
                j = jj
                if jj >= i:
                    j = jj + 1
                for k2 in range(nray):
                    aidx = i * (ns - 1) * (nray + 1) + jj * (nray + 1) + k2
                    if aux_valid[w, aidx] != 0:
                        j0 = j * nray + k2
                        d0 = dist[w, j0]
                        if d0 > 0.0:
                            q = pnt[w, j0] + vec[w, j0] * d0
                            vx, vy, vz = q[0] - rx[0], q[1] - rx[1], q[2] - rx[2]
                            di = wp.sqrt(vx * vx + vy * vy + vz * vz)
                            if di > 1e-8:
                                ux, uy, uz = vx / di, vy / di, vz / di
                                th_rx = wp.acos(wp.clamp(ux * axis[0] + uy * axis[1] + uz * axis[2], -1.0, 1.0))
                                cos_out = -(normal[w, j0][0] * ux + normal[w, j0][1] * uy + normal[w, j0][2] * uz)
                                if th_rx <= rx_half(di, sensor_near, sensor_far,
                                                   sensor_near_dist, sensor_far_dist) and cos_out > 0.0:
                                    vis = adist[w, aidx]
                                    if vis > 0.0 and wp.abs(vis - di) < 1e-3:
                                        g = geomid[w, j0]
                                        if g >= 0:
                                            amb[w, i] += ray_power[k2] * albedo[g] * cos_out / (wp.pi * di * di)
                aidx = i * (ns - 1) * (nray + 1) + jj * (nray + 1) + nray
                if aux_valid[w, aidx] != 0:
                    tx = origin[w, j]
                    vx, vy, vz = tx[0] - rx[0], tx[1] - rx[1], tx[2] - rx[2]
                    di = wp.sqrt(vx * vx + vy * vy + vz * vz)
                    if di > 1e-8:
                        ux, uy, uz = vx / di, vy / di, vz / di
                        th_rx = wp.acos(wp.clamp(ux * axis[0] + uy * axis[1] + uz * axis[2], -1.0, 1.0))
                        th_tx = wp.acos(wp.clamp(-(ux * (rot[w, j] @ wp.vec3f(1.0, 0.0, 0.0))[0] +
                                                  uy * (rot[w, j] @ wp.vec3f(1.0, 0.0, 0.0))[1] +
                                                  uz * (rot[w, j] @ wp.vec3f(1.0, 0.0, 0.0))[2]), -1.0, 1.0))
                        if th_rx <= rx_half(di, sensor_near, sensor_far,
                                           sensor_near_dist, sensor_far_dist) and th_tx <= 3.0 * foi_half:
                            if ageom[w, aidx] == housing[j]:
                                amb[w, i] += emission_weight(th_tx, foi_half) / (di * di)
        acc_yaw[w, i] += bearing[w, i]
        acc_beam[w, i] += beam_yaw[w, i]
        acc_n[w, i] += 1


@wp.kernel
def tracker_from_readings(x: wp.array2d(dtype=wp.float32),
                          P: wp.array2d(dtype=wp.float32),
                          ok: wp.array(dtype=wp.int32),
                          tlast: wp.array(dtype=wp.float32),
                          finished: wp.array2d(dtype=wp.int32),
                          rd_valid: wp.array2d(dtype=wp.int32),
                          rd_range: wp.array2d(dtype=wp.float32),
                          rd_sigma: wp.array2d(dtype=wp.float32),
                          rd_yaw: wp.array2d(dtype=wp.float32),
                          rd_slot: wp.array2d(dtype=wp.int32),
                          rd_time: wp.array2d(dtype=wp.float32),
                          corner_x: wp.array(dtype=wp.float32),
                          corner_y: wp.array(dtype=wp.float32),
                          bg: wp.array2d(dtype=wp.float32),
                          R_ball: float, limit: float, lost_after: float,
                          range_floor: float, bg_margin: float, gate: float,
                          init_pos: float, init_vel: float,
                          sensor_near: float, sensor_far: float,
                          sensor_near_dist: float, sensor_far_dist: float):
    w = wp.tid()
    ns = finished.shape[1]
    for s in range(ns):
        if finished[w, s] != 0:
            tracker_update(w, s, rd_valid[w, s], rd_range[w, s], rd_sigma[w, s],
                           rd_yaw[w, s], rd_time[w, s], rd_slot[w, s],
                           corner_x, corner_y, bg, x, P, ok, tlast,
                           R_ball, limit, lost_after, range_floor, bg_margin,
                           gate, init_pos, init_vel, sensor_near, sensor_far,
                           sensor_near_dist, sensor_far_dist)
            finished[w, s] = 0
