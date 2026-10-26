"""Analytic prior for the edge-coupled (parallel-coupled) microstrip band-pass filter.

Physics (Pozar, Microwave Engineering 4th ed., ch. 8.7 "Coupled line filters"; Matthaei, Young & Jones 1980,
"Microwave Filters, Impedance-Matching Networks, and Coupling Structures", secs 4.05 and 8.09):
  * N = 3 Chebyshev prototype -> N+1 = 4 coupled-line sections. Sections 1 and 4 share (w1, g1, l1), sections 2
    and 3 share (w2, g2, l2). Each section has its own strip width, so the Pozar Z0e/Z0o targets are met exactly.
  * Each section is the exact two-port of a symmetric coupled-line pair with the two far ends open (Pozar 8.102-8.104,
    generalised to different even/odd mode phase velocities): Z11 = -j/2 (Z0e cot th_e + Z0o cot th_o),
    Z21 = -j/2 (Z0e csc th_e - Z0o csc th_o). Sections are cascaded as ABCD matrices and converted to S (50 ohm).
  * Even/odd mode impedances and effective permittivities of coupled microstrip: Kirschning & Jansen (IEEE MTT-32,
    1984) static equations, built on the Hammerstad & Jensen (1980) single-line equations. Zero copper thickness.
  * Open-end extension (Hammerstad) added to each section length; dielectric loss via the filling factor; copper is
    lossless (Palace models the copper as a PEC sheet, so the prior does the same). Optional Kobayashi dispersion.
  * The width step between a w1 part and a w2 part of a resonator is not modelled (both parts share the strip axis).

Parameter vector p = (w1, w2, g1, g2, l1, l2, eps_r, h): lengths in mm, f in GHz.

    S11, S21 = s_prior(p, f_ghz)

This prior is the "S_prior" term of the learned model S = S_prior + NN(p, f). It is not meant to match the FEM; it
is meant to carry the resonant structure so the network learns a small correction.
"""
from __future__ import annotations

import numpy as np

C0 = 299792458.0            # m/s
ETA0 = 376.730313668        # ohm
PARAM_NAMES = ("w1", "w2", "g1", "g2", "l1", "l2", "eps_r", "h")

# substrate presets: eps_r, h (mm), tan delta
SUBSTRATES = {
    "RO4350B": {"eps_r": 3.66, "h": 0.508, "tand": 0.0037},
    "FR4": {"eps_r": 4.4, "h": 1.6, "tand": 0.02},
    "alumina": {"eps_r": 9.8, "h": 0.635, "tand": 0.0001},
}


def tand_for(eps_r: float) -> float:
    """Loss tangent of the preset substrate whose eps_r is closest (the parameter vector carries no tan delta)."""
    best = min(SUBSTRATES.values(), key=lambda s: abs(s["eps_r"] - eps_r))
    return best["tand"]


# ----------------------------------------------------------------------------- prototype and inverters

def chebyshev_g(n: int, ripple_db: float):
    """Element values g1..gN, g_{N+1} of the equal-ripple low-pass prototype, closed-form recurrence.

    Matthaei, Young & Jones (1980) sec. 4.05 eqs (4.05-1)/(4.05-2); Pozar 4th ed. eqs (8.72) and Table 8.4:
      beta = ln coth(L_Ar / 17.37),  gamma = sinh(beta / 2N),
      a_k = sin((2k-1) pi / 2N),  b_k = gamma^2 + sin^2(k pi / N),
      g1 = 2 a_1 / gamma,  g_k = 4 a_{k-1} a_k / (b_{k-1} g_{k-1}),  g_{N+1} = 1 (N odd) or coth^2(beta/4) (N even).
    N = 3, 0.5 dB: 1.5963, 1.0967, 1.5963, 1.0000 (checked in tests/test_prior.py).
    """
    beta = np.log(1 / np.tanh(ripple_db / (40 / np.log(10))))     # 40/ln 10 = 17.3718, the "17.37" of the textbooks
    gamma = np.sinh(beta / (2 * n))
    a = [np.sin((2 * k - 1) * np.pi / (2 * n)) for k in range(1, n + 1)]
    b = [gamma ** 2 + np.sin(k * np.pi / n) ** 2 for k in range(1, n + 1)]
    g = [2 * a[0] / gamma]
    for k in range(2, n + 1):
        g.append(4 * a[k - 2] * a[k - 1] / (b[k - 2] * g[-1]))
    g.append(1.0 if n % 2 else 1 / np.tanh(beta / 4) ** 2)
    return [float(v) for v in g]


def coupled_line_impedances(n=3, ripple_db=0.5, fbw=0.10, z0=50.0):
    """(Z0e, Z0o) of the N+1 coupled-line sections (Pozar 4th ed. eqs 8.121 and 8.108):
      Z0 J1 = sqrt(pi D / (2 g1)),  Z0 Jk = pi D / (2 sqrt(g_{k-1} g_k)),  Z0 J_{N+1} = sqrt(pi D / (2 g_N g_{N+1})),
      Z0e = Z0 [1 + J Z0 + (J Z0)^2],  Z0o = Z0 [1 - J Z0 + (J Z0)^2].
    Pozar Example 8.8 (N = 3, 0.5 dB, 10 %): 70.61/39.24 and 56.64/44.77 ohm (checked in tests).
    Returns ([(Z0e, Z0o), ...], g, [J Z0, ...]).
    """
    g = chebyshev_g(n, ripple_db)
    jz = [np.sqrt(np.pi * fbw / (2 * g[0]))]
    jz += [np.pi * fbw / (2 * np.sqrt(g[k - 1] * g[k])) for k in range(1, n)]
    jz.append(np.sqrt(np.pi * fbw / (2 * g[n - 1] * g[n])))
    return [(z0 * (1 + j + j * j), z0 * (1 - j + j * j)) for j in jz], g, [float(v) for v in jz]


# ----------------------------------------------------------------------------- single microstrip (Hammerstad-Jensen)

def _hj_a(u):
    return 1 + np.log((u ** 4 + (u / 52) ** 2) / (u ** 4 + 0.432)) / 49 + np.log(1 + (u / 18.1) ** 3) / 18.7


def _hj_b(er):
    return 0.564 * ((er - 0.9) / (er + 3)) ** 0.053


def microstrip(w, h, eps_r):
    """Static Z0 (ohm) and effective permittivity of a zero-thickness microstrip (Hammerstad & Jensen 1980)."""
    u = w / h
    fu = 6 + (2 * np.pi - 6) * np.exp(-((30.666 / u) ** 0.7528))
    z01 = ETA0 / (2 * np.pi) * np.log(fu / u + np.sqrt(1 + (2 / u) ** 2))      # impedance in air
    eeff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 10 / u) ** (-_hj_a(u) * _hj_b(eps_r))
    return z01 / np.sqrt(eeff), eeff


def open_end_extension(w, h, eps_r):
    """Hammerstad open-end length extension (mm)."""
    _, eeff = microstrip(w, h, eps_r)
    u = w / h
    return 0.412 * h * (eeff + 0.3) / (eeff - 0.258) * (u + 0.262) / (u + 0.813)


def kobayashi_eeff(f_hz, eeff0, eps_r, w, h):
    """Frequency-dependent effective permittivity (Kobayashi 1988) for a mode with static value eeff0."""
    u = w / h
    h_m = h * 1e-3
    fk = C0 * np.arctan(eps_r * np.sqrt((eeff0 - 1) / (eps_r - eeff0))) / (2 * np.pi * h_m * np.sqrt(eps_r - eeff0))
    f50 = fk / (0.75 + (0.75 - 0.332 / eps_r ** 1.73) * u)
    m0 = 1 + 1 / (1 + np.sqrt(u)) + 0.32 / (1 + np.sqrt(u)) ** 3
    mc = 1 + 1.4 / (1 + u) * (0.15 - 0.235 * np.exp(-0.45 * f_hz / f50)) if u <= 0.7 else 1.0
    m = np.minimum(m0 * mc, 2.32)
    return eps_r - (eps_r - eeff0) / (1 + (f_hz / f50) ** m)


# ----------------------------------------------------------------------------- coupled microstrip (Kirschning-Jansen)

def coupled_microstrip(w, s, h, eps_r):
    """Static even/odd mode impedances and effective permittivities (Kirschning & Jansen 1984, zero thickness).

    Returns Z0e, Z0o (ohm), eps_e, eps_o.
    """
    u, g = w / h, s / h
    z0, eeff = microstrip(w, h, eps_r)
    z01 = z0 * np.sqrt(eeff)                                           # single line in air
    v = u * (20 + g ** 2) / (10 + g ** 2) + g * np.exp(-g)
    eps_e = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 10 / v) ** (-_hj_a(v) * _hj_b(eps_r))
    a0 = 0.7287 * (eeff - (eps_r + 1) / 2) * (1 - np.exp(-0.179 * u))
    b0 = 0.747 * eps_r / (0.15 + eps_r)
    c0 = b0 - (b0 - 0.207) * np.exp(-0.414 * u)
    d0 = 0.593 + 0.694 * np.exp(-0.562 * u)
    eps_o = ((eps_r + 1) / 2 + a0 - eeff) * np.exp(-c0 * g ** d0) + eeff
    q1 = 0.8695 * u ** 0.194
    q2 = 1 + 0.7519 * g + 0.189 * g ** 2.31
    q3 = 0.1975 + (16.6 + (8.4 / g) ** 6) ** (-0.387) + np.log(g ** 10 / (1 + (g / 3.4) ** 10)) / 241
    q4 = 2 * q1 / q2 / (u ** q3 * np.exp(-g) + (2 - np.exp(-g)) * u ** (-q3))
    z0e = z01 / np.sqrt(eps_e) / (1 - z01 / ETA0 * q4)
    q5 = 1.794 + 1.14 * np.log(1 + 0.638 / (g + 0.517 * g ** 2.43))
    q6 = 0.2305 + np.log(g ** 10 / (1 + (g / 5.8) ** 10)) / 281.3 + np.log(1 + 0.598 * g ** 1.154) / 5.1
    q7 = (10 + 190 * g ** 2) / (1 + 82.3 * g ** 3)
    q8 = np.exp(-6.5 - 0.95 * np.log(g) - (g / 0.15) ** 5)
    q9 = np.log(q7) * (q8 + 1 / 16.5)
    q10 = (q2 * q4 - q5 * np.exp(np.log(u) * q6 * u ** (-q9))) / q2
    z0o = z01 / np.sqrt(eps_o) / (1 - z01 / ETA0 * q10)
    return float(z0e), float(z0o), float(eps_e), float(eps_o)


# ----------------------------------------------------------------------------- circuit model

def _mode_theta(f_hz, l_mm, eeff, eps_r, tand):
    """Complex electrical length theta = (beta - j alpha) l for one mode (dielectric loss via the filling factor)."""
    beta = 2 * np.pi * f_hz / C0 * np.sqrt(eeff)
    q = eps_r * (eeff - 1) / (eeff * (eps_r - 1)) if eps_r > 1 else 0.0     # filling factor
    alpha = beta * q * tand / 2
    return (beta - 1j * alpha) * l_mm * 1e-3


def section_abcd(f_hz, z0e, z0o, th_e, th_o):
    """ABCD matrices (M,2,2) of one coupled-line band-pass section (far ends open), exact TEM coupled-line model."""
    z11 = -0.5j * (z0e / np.tan(th_e) + z0o / np.tan(th_o))
    z21 = -0.5j * (z0e / np.sin(th_e) - z0o / np.sin(th_o))
    abcd = np.empty((np.size(f_hz), 2, 2), dtype=complex)
    abcd[:, 0, 0] = z11 / z21
    abcd[:, 0, 1] = (z11 ** 2 - z21 ** 2) / z21
    abcd[:, 1, 0] = 1 / z21
    abcd[:, 1, 1] = z11 / z21
    return abcd


def abcd_to_s(abcd, z0=50.0):
    a, b, c, d = abcd[:, 0, 0], abcd[:, 0, 1], abcd[:, 1, 0], abcd[:, 1, 1]
    den = a + b / z0 + c * z0 + d
    s11 = (a + b / z0 - c * z0 - d) / den
    s21 = 2 / den
    s12 = 2 * (a * d - b * c) / den
    s22 = (-a + b / z0 - c * z0 + d) / den
    return s11, s21, s12, s22


def unpack(p):
    p = np.asarray(p, dtype=float).reshape(-1)
    if p.size != 8:
        raise ValueError(f"p must have 8 entries (w1, w2, g1, g2, l1, l2, eps_r, h), got {p.size}")
    return [float(x) for x in p]


def filter_sections(p, tand=None):
    """Per-section (w, Z0e, Z0o, eps_e, eps_o, length_mm incl. open-end extension) for p = (w1, w2, g1, g2, l1, l2,
    eps_r, h): sections 1, 4 use (w1, g1, l1), sections 2, 3 use (w2, g2, l2)."""
    w1, w2, g1, g2, l1, l2, eps_r, h = unpack(p)
    out = []
    for w, g, l in ((w1, g1, l1), (w2, g2, l2), (w2, g2, l2), (w1, g1, l1)):
        z0e, z0o, ee, eo = coupled_microstrip(w, g, h, eps_r)
        out.append((w, z0e, z0o, ee, eo, l + open_end_extension(w, h, eps_r)))
    return out


def s_prior(p, f_ghz, tand=None, dispersion=True, z0=50.0):
    """S11, S21 (complex arrays, shape (M,)) of the prior at frequencies f_ghz for design p (mm)."""
    w1, w2, g1, g2, l1, l2, eps_r, h = unpack(p)
    f = np.atleast_1d(np.asarray(f_ghz, dtype=float)) * 1e9
    if tand is None:
        tand = tand_for(eps_r)
    total = np.tile(np.eye(2, dtype=complex), (f.size, 1, 1))
    for w, z0e, z0o, ee, eo, l in filter_sections(p, tand):
        if dispersion:
            ee_f, eo_f = kobayashi_eeff(f, ee, eps_r, w, h), kobayashi_eeff(f, eo, eps_r, w, h)
            z0e_f, z0o_f = z0e * np.sqrt(ee / ee_f), z0o * np.sqrt(eo / eo_f)   # air-line impedance held fixed
        else:
            ee_f, eo_f, z0e_f, z0o_f = ee, eo, z0e, z0o
        abcd = section_abcd(f, z0e_f, z0o_f, _mode_theta(f, l, ee_f, eps_r, tand), _mode_theta(f, l, eo_f, eps_r, tand))
        total = total @ abcd
    s11, s21, _, _ = abcd_to_s(total, z0)
    return s11, s21


# ----------------------------------------------------------------------------- response metrics

def passband_metrics(f_ghz, s21, drop_db=3.0):
    """f0 (centre of the -3 dB band, GHz), BW (GHz), IL (min insertion loss in dB, positive) from |S21|.

    The -3 dB band is the widest contiguous run around the |S21| maximum that stays within drop_db of it.
    Returns NaNs when there is no band inside the grid.
    """
    f = np.asarray(f_ghz, float)
    mag = 20 * np.log10(np.abs(s21) + 1e-30)
    k = int(np.argmax(mag))
    thr = mag[k] - drop_db
    lo = k
    while lo > 0 and mag[lo - 1] >= thr:
        lo -= 1
    hi = k
    while hi < f.size - 1 and mag[hi + 1] >= thr:
        hi += 1
    if lo == 0 or hi == f.size - 1:
        return float("nan"), float("nan"), float(-mag[k])

    def edge(i, j):      # linear interpolation of the crossing between samples i (inside) and j (outside)
        return f[i] + (thr - mag[i]) * (f[j] - f[i]) / (mag[j] - mag[i])

    f_lo, f_hi = edge(lo, lo - 1), edge(hi, hi + 1)
    return float(0.5 * (f_lo + f_hi)), float(f_hi - f_lo), float(-mag[k])


def _parabolic_extremum(f, y, k):
    """Refine the position/value of an extremum at sample k with a parabola through k-1, k, k+1."""
    if k <= 0 or k >= y.size - 1:
        return float(f[k]), float(y[k])
    y0, y1, y2 = y[k - 1], y[k], y[k + 1]
    den = y0 - 2 * y1 + y2
    if abs(den) < 1e-12:
        return float(f[k]), float(y[k])
    d = 0.5 * (y0 - y2) / den
    return float(f[k] + d * (f[k + 1] - f[k])), float(y1 - 0.25 * (y0 - y2) * d)


def passband_features(f_ghz, s11, s21, n_zeros=3):
    """Equiripple diagnostics from |S11| inside the 3 dB band of |S21|.

    Returns dict: f0, BW, IL (from passband_metrics), zeros_GHz (the n deepest |S11| minima in the band, sorted),
    zero_depth_dB, peaks_dB (|S11| maxima between consecutive zeros), peak_spread_dB (max - min of peaks),
    rl_band_dB (worst |S11| between the outer zeros).
    """
    f = np.asarray(f_ghz, float)
    f0, bw, il = passband_metrics(f, s21)
    out = {"f0_GHz": f0, "BW_GHz": bw, "IL_dB": il, "zeros_GHz": [], "zero_depth_dB": [], "peaks_dB": [],
           "peak_spread_dB": float("nan"), "rl_band_dB": float("nan")}
    if not np.isfinite(f0):
        return out
    s11_db = 20 * np.log10(np.abs(s11) + 1e-30)
    band = np.where((f >= f0 - 0.6 * bw) & (f <= f0 + 0.6 * bw))[0]
    if band.size < 5:
        return out
    seg = s11_db[band]
    mins = [i for i in range(1, seg.size - 1) if seg[i] < seg[i - 1] and seg[i] <= seg[i + 1]]
    mins = sorted(mins, key=lambda i: seg[i])[:n_zeros]
    mins.sort()
    zeros = [_parabolic_extremum(f[band], seg, i) for i in mins]
    out["zeros_GHz"] = [z[0] for z in zeros]
    out["zero_depth_dB"] = [z[1] for z in zeros]
    peaks = []
    for a, b in zip(mins[:-1], mins[1:]):
        k = a + int(np.argmax(seg[a:b + 1]))
        peaks.append(_parabolic_extremum(f[band], seg, k)[1])
    out["peaks_dB"] = peaks
    if peaks:
        out["peak_spread_dB"] = float(max(peaks) - min(peaks))
    if len(mins) >= 2:
        out["rl_band_dB"] = float(-seg[mins[0]:mins[-1] + 1].max())
    return out


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    d = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else "design.json").read_text())
    p = d["nominal"]["p"]
    f = np.arange(2.0, 8.0001, 0.01)
    s11, s21 = s_prior(p, f)
    print(f"p = {p}\nsections (w, Z0e, Z0o, eps_e, eps_o, l_eff mm):")
    for sec in filter_sections(p):
        print("  " + ", ".join(f"{x:.3f}" for x in sec))
    print(json.dumps(passband_features(f, s11, s21), indent=1))
