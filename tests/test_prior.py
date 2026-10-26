"""Checks of prior.py against scikit-rf, the Matthaei-Young-Jones recurrence and Pozar's textbook example.

    python -m pytest tests
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import skrf
from skrf.network import connect

import design  # noqa: E402
import prior  # noqa: E402


@pytest.mark.parametrize("w,h,er", [(1.1, 0.508, 3.66), (3.0, 1.6, 4.4), (0.6, 0.635, 9.8), (0.2, 0.508, 3.66)])
def test_single_microstrip_matches_skrf_hammerstad(w, h, er):
    z, e = prior.microstrip(w, h, er)
    m = skrf.media.MLine(frequency=skrf.Frequency(1, 1, 1, "ghz"), w=w * 1e-3, h=h * 1e-3, t=0, ep_r=er,
                         model="hammerstadjensen", disp="none")
    assert z == pytest.approx(float(m.z0_characteristic[0].real), rel=1e-4)
    assert e == pytest.approx(float(m.ep_reff_f[0].real), rel=1e-4)


def test_coupled_modes_ordered_and_converge_to_single_line():
    z0, eeff = prior.microstrip(1.1, 0.508, 3.66)
    prev_k = 1.0
    for g in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0):
        ze, zo, ee, eo = prior.coupled_microstrip(1.1, g, 0.508, 3.66)
        assert ze > z0 > zo and ee > eo
        k = (ze - zo) / (ze + zo)
        assert k < prev_k           # coupling falls monotonically with the gap
        prev_k = k
    ze, zo, ee, eo = prior.coupled_microstrip(1.1, 20.0, 0.508, 3.66)
    assert ze == pytest.approx(z0, rel=2e-3) and zo == pytest.approx(z0, rel=2e-3)
    assert ee == pytest.approx(eeff, rel=2e-3) and eo == pytest.approx(eeff, rel=2e-3)


@pytest.mark.parametrize("n,ripple,expect", [
    (3, 0.5, [1.5963, 1.0967, 1.5963, 1.0000]),                    # MYJ Table 4.05-2(a); Pozar Table 8.4
    (3, 3.0, [3.3487, 0.7117, 3.3487, 1.0000]),
    (4, 0.5, [1.6703, 1.1926, 2.3661, 0.8419, 1.9841]),
    (5, 0.5, [1.7058, 1.2296, 2.5408, 1.2296, 1.7058, 1.0000]),
    (2, 0.5, [1.4029, 0.7071, 1.9841]),
])
def test_chebyshev_prototype_recurrence_matches_tables(n, ripple, expect):
    """Closed-form recurrence (MYJ sec. 4.05, Pozar eqs 8.72) against the published tables."""
    g = prior.chebyshev_g(n, ripple)
    assert g == pytest.approx(expect, abs=2e-4)


def test_chebyshev_prototype_ripple_from_prototype_response():
    """The N=3, 0.5 dB prototype, as a lumped ladder, has exactly 0.5 dB ripple up to omega = 1."""
    g = prior.chebyshev_g(3, 0.5)
    w = np.linspace(0.01, 1.0, 2000)
    # ladder: series L = g1, shunt C = g2, series L = g3, load g4 (low-pass prototype, source 1 ohm)
    s21 = []
    for om in w:
        abcd = np.eye(2, dtype=complex)
        for k, val in enumerate(g[:3]):
            if k % 2 == 0:
                m = np.array([[1, 1j * om * val], [0, 1]])
            else:
                m = np.array([[1, 0], [1j * om * val, 1]])
            abcd = abcd @ m
        a, b, c, d = abcd.ravel()
        rl = g[3]
        s21.append(2 * np.sqrt(rl) / (a * rl + b + c * rl + d))
    db = 20 * np.log10(np.abs(s21))
    assert db.max() == pytest.approx(0.0, abs=1e-3)
    assert db.min() == pytest.approx(-0.5, abs=2e-3)


def test_pozar_example_8_8_impedances():
    """Pozar, Example 8.8 (4th ed.): N=3, 0.5 dB ripple, 10 % bandwidth -> Z0e/Z0o 70.61/39.24 and 56.64/44.77."""
    zz, g, jz = prior.coupled_line_impedances(3, 0.5, 0.10, 50.0)
    assert g[:3] == pytest.approx([1.5963, 1.0967, 1.5963], abs=2e-4) and g[3] == pytest.approx(1.0)
    assert jz == pytest.approx([0.3137, 0.1187, 0.1187, 0.3137], abs=1e-4)
    assert zz[0] == pytest.approx((70.61, 39.24), abs=0.02)
    assert zz[1] == pytest.approx((56.64, 44.77), abs=0.02)
    assert zz[2] == zz[1] and zz[3] == zz[0]


def _ideal_tem_filter(f_ghz, zz, f0=2.0):
    """Prior circuit with ideal TEM sections (eps_e = eps_o = 1, lossless, quarter wave at f0)."""
    f = f_ghz * 1e9
    total = np.tile(np.eye(2, dtype=complex), (f.size, 1, 1))
    l_m = prior.C0 / (4 * f0 * 1e9)
    for ze, zo in zz:
        th = 2 * np.pi * f / prior.C0 * l_m
        total = total @ prior.section_abcd(f, ze, zo, th, th)
    s11, s21, _, _ = prior.abcd_to_s(total)
    return s11, s21


def test_pozar_example_8_8_response():
    """Equal-ripple passband at 2 GHz: |S21| within 0.5 dB ripple over most of the 10 % band, > 20 dB out of band."""
    zz, _, _ = prior.coupled_line_impedances(3, 0.5, 0.10, 50.0)
    f = np.linspace(1.5, 2.5, 2001)
    _, s21 = _ideal_tem_filter(f, zz)
    db = 20 * np.log10(np.abs(s21))
    f0, bw, il = prior.passband_metrics(f, s21)
    assert il < 0.01                                   # lossless: 0 dB at the ripple maxima
    assert f0 == pytest.approx(2.0, abs=0.01)
    assert 0.09 < bw / f0 < 0.125                      # 3 dB bandwidth close to the 10 % design (ripple band is 0.5 dB)
    inband = (f > 1.91) & (f < 2.09)
    assert db[inband].min() > -0.6                     # 0.5 dB ripple spec (coupled-line approx. of the prototype)
    assert db[f < 1.6].max() < -20 and db[f > 2.4].max() < -20


def test_sections_match_skrf_even_odd_superposition():
    """Build each coupled-line section in scikit-rf from even/odd mode TEM lines (4-port superposition, open far
    ends, cascade) and compare with the ABCD implementation."""
    zz, _, _ = prior.coupled_line_impedances(3, 0.5, 0.10, 50.0)
    freq = skrf.Frequency(1.5, 2.5, 201, "ghz")
    f = freq.f
    l_m = prior.C0 / (4 * 2.0e9)
    ntw = None
    for ze, zo in zz:
        se = skrf.media.DefinedGammaZ0(freq, z0=ze, gamma=1j * 2 * np.pi * f / prior.C0).line(l_m, "m")
        so = skrf.media.DefinedGammaZ0(freq, z0=zo, gamma=1j * 2 * np.pi * f / prior.C0).line(l_m, "m")
        se.renormalize(50); so.renormalize(50)
        s4 = np.zeros((f.size, 4, 4), dtype=complex)
        # ports: 0 = strip a left, 1 = strip a right, 2 = strip b left, 3 = strip b right
        s4[:, :2, :2] = s4[:, 2:, 2:] = 0.5 * (se.s + so.s)
        s4[:, :2, 2:] = s4[:, 2:, :2] = 0.5 * (se.s - so.s)
        n4 = skrf.Network(frequency=freq, s=s4, z0=50)
        med = skrf.media.DefinedGammaZ0(freq, z0=50)
        n = connect(n4, 1, med.open(), 0)         # strip a right end open -> ports (a left, b left, b right)
        n = connect(n, 1, med.open(), 0)          # strip b left end open  -> ports (a left, b right)
        ntw = n if ntw is None else ntw ** n
    s11_ref, s21_ref = _ideal_tem_filter(f / 1e9, zz)
    assert np.allclose(ntw.s[:, 0, 0], s11_ref, atol=1e-8)
    assert np.allclose(ntw.s[:, 1, 0], s21_ref, atol=1e-8)


def test_synthesis_hits_targets_and_prior_is_equiripple():
    """Per-section widths: Z0e/Z0o of both section types match Pozar exactly; the lossless prior has three
    reflection zeros and three equal |S11| peaks at the 0.5 dB ripple level (-9.64 dB)."""
    sub = prior.SUBSTRATES["RO4350B"]
    p, rec = design.synthesize(sub["eps_r"], sub["h"])
    for got, want in zip(rec["Z0e_Z0o_microstrip"], rec["Z0e_Z0o_target"]):
        assert got == pytest.approx(want, abs=1e-3)
    f = np.round(np.arange(4.0, 6.0001, 0.002), 4)
    s11, s21 = prior.s_prior(p, f, tand=0.0)
    feat = prior.passband_features(f, s11, s21)
    assert feat["f0_GHz"] == pytest.approx(5.0, abs=0.02) and 0.09 < feat["BW_GHz"] / feat["f0_GHz"] < 0.13
    assert len(feat["zeros_GHz"]) == 3 and len(feat["peaks_dB"]) == 2
    ripple_level = 10 * np.log10(1 - 10 ** (-0.05))          # |S11|^2 at a 0.5 dB |S21| dip: -9.64 dB
    for pk in feat["peaks_dB"]:
        assert pk == pytest.approx(ripple_level, abs=0.6)   # microstrip sections are not ideal TEM: allow 0.6 dB
    assert feat["peak_spread_dB"] < 1.0
    assert np.all(np.abs(s11) ** 2 + np.abs(s21) ** 2 <= 1 + 1e-9)


def test_prior_with_loss_is_passive():
    sub = prior.SUBSTRATES["RO4350B"]
    p, _ = design.synthesize(sub["eps_r"], sub["h"])
    s11, s21 = prior.s_prior(p, design.F_GRID)
    assert s11.shape == s21.shape == design.F_GRID.shape
    assert np.all(np.abs(s11) ** 2 + np.abs(s21) ** 2 <= 1 + 1e-9)
    f0, bw, il = prior.passband_metrics(design.F_GRID, s21)
    assert f0 == pytest.approx(5.0, abs=0.02) and il < 1.5
