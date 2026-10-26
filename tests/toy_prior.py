"""Test-only analytic band-pass model.  NOT the project prior (that is families/<family_id>/prior.py).

A lossless 2nd-order Butterworth band-pass whose centre frequency and
fractional bandwidth depend smoothly on p = (w, g1, g2, l1, l2, eps_r, h):

    f0  = 299.79 / (4 (l1 + l2) sqrt(eps_r))   GHz   (l in mm)
    fbw = 0.15 + 0.15 (g1 + g2) / (2 w)
    x   = (f/f0 - f0/f) / fbw
    S21 = 1 / (1 - x^2 + j sqrt2 x),   S11 = -x^2 / (1 - x^2 + j sqrt2 x)

|S11|^2 + |S21|^2 = 1 exactly, so the passivity penalty must be zero on it.
`toy_truth` adds a known smooth perturbation (resonator lengths 1.5 % longer
and a Gaussian dip in |S21| of depth that scales with eps_r) to stand in for
the Palace result in the synthetic end-to-end test.
"""
import numpy as np

BOX = {"w": (0.8, 1.6), "g1": (0.15, 0.5), "g2": (0.3, 0.9), "l1": (4.0, 7.0), "l2": (4.0, 7.0),
       "eps_r": (3.0, 4.5), "h": (0.5, 1.0)}


def toy_prior(p, f_ghz):
    w, g1, g2, l1, l2, eps_r, h = np.asarray(p, dtype=float)
    f = np.asarray(f_ghz, dtype=float)
    f0 = 299.79 / (4.0 * (l1 + l2) * np.sqrt(eps_r))
    fbw = 0.15 + 0.15 * (g1 + g2) / (2.0 * w)
    x = (f / f0 - f0 / f) / fbw
    den = 1.0 - x ** 2 + 1j * np.sqrt(2.0) * x
    s21 = 1.0 / den
    s11 = -(x ** 2) / den
    return s11, s21


def toy_truth(p, f_ghz, dip=0.08):
    """Prior + known smooth perturbation (plays the role of the FEM result)."""
    p = np.asarray(p, dtype=float).copy()
    p[3] *= 1.015; p[4] *= 1.015
    s11, s21 = toy_prior(p, f_ghz)
    f = np.asarray(f_ghz, dtype=float)
    f0 = 299.79 / (4.0 * (p[3] + p[4]) * np.sqrt(p[5]))
    g = np.exp(-((f - f0) / (0.06 * f0)) ** 2)
    a = dip * (p[5] - 3.0) / 1.5                 # depth grows with eps_r
    s21 = s21 * (1.0 - a * g)
    # keep it exactly passive: scale S11 so |S11|^2 + |S21|^2 <= 1
    pw = np.abs(s11) ** 2 + np.abs(s21) ** 2
    s11 = np.where(pw > 1, s11 * np.sqrt(np.maximum(1 - np.abs(s21) ** 2, 0) / np.maximum(np.abs(s11) ** 2, 1e-30)), s11)
    return s11, s21


def sample_box(n, rng, box=BOX):
    names = ("w", "g1", "g2", "l1", "l2", "eps_r", "h")
    return np.stack([rng.uniform(*box[k], size=n) for k in names], axis=1)
