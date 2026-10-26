"""Unit tests: shapes, reciprocity, passivity penalty, prior_wrap, synthetic end-to-end.

Run:  python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import net as N                                   # noqa: E402
import prior_wrap as PW                           # noqa: E402
from trainer import TensorSet, fit, predict       # noqa: E402
from toy_prior import toy_prior, toy_truth, sample_box  # noqa: E402

torch.set_num_threads(4)


def _toy_batch(P, f):
    return PW.s_prior_batch(P, f, fn=toy_prior)


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_params", [7, 8, 6])
def test_shapes_and_parameter_count(n_params):
    """The branch input size follows the data (param_names), not a constant."""
    m = N.DeepONetCorrection(n_params=n_params, use_prior=False)
    p = torch.randn(5, n_params); f = torch.linspace(1.0, 5.0, 33)
    m.p_std.fit(p); m.f_norm.fit(f)
    out = m(p, f)
    assert out.shape == (5, 4, 33)
    assert m.branch(p).shape == (5, 4, 64)
    assert m.basis(f).shape == (33, 64)
    assert m.basis(f, p).shape == (5, 33, 64)          # per-design warped basis
    n = m.n_parameters()
    assert 30_000 < n < 80_000, n
    s11, s21 = N.unpack(out)
    assert s11.shape == (5, 33) and s11.is_complex()
    mlp = N.PlainMLP(n_freq=33, n_params=n_params)
    assert mlp(p).shape == (5, 4, 33)
    with pytest.raises(RuntimeError):
        m(torch.randn(5, n_params + 1), f)             # wrong design-vector length must fail loudly


def test_reciprocity_and_symmetry():
    m = N.DeepONetCorrection(n_params=7, use_prior=False)
    p = torch.randn(3, 7); f = torch.linspace(1.0, 5.0, 17)
    S = N.to_2x2(m(p, f))
    assert S.shape == (3, 17, 2, 2)
    assert torch.allclose(S[..., 0, 1], S[..., 1, 0])      # S12 == S21
    assert torch.allclose(S[..., 0, 0], S[..., 1, 1])      # S22 == S11
    # prior ON path keeps the property too
    m.switch_prior_on()
    sp = torch.randn(3, 4, 17)
    S = N.to_2x2(m(p, f, sp))
    assert torch.allclose(S[..., 0, 1], S[..., 1, 0]) and torch.allclose(S[..., 0, 0], S[..., 1, 1])


def test_passivity_penalty():
    f = np.linspace(2.0, 6.0, 101)
    rng = np.random.default_rng(0)
    P = sample_box(8, rng)
    s = torch.as_tensor(_toy_batch(P, f))
    assert N.passivity_penalty(s).item() == pytest.approx(0.0, abs=1e-6)     # lossless -> exactly passive
    s_bad = s.clone(); s_bad[:, 2] *= 1.5                                    # |S21| x1.5 -> active
    assert N.passivity_penalty(s_bad).item() > 0.0
    # penalty gradient points back toward passivity
    s_bad.requires_grad_(True)
    N.passivity_penalty(s_bad).backward()
    assert (s_bad.grad[:, 2] * s_bad[:, 2]).sum().item() > 0


def test_switch_prior_on_returns_exact_prior():
    m = N.DeepONetCorrection(n_params=7, use_prior=False)
    p = torch.randn(4, 7); f = torch.linspace(1.0, 5.0, 21)
    m.p_std.fit(p); m.f_norm.fit(f)
    sp = torch.randn(4, 4, 21)
    m.switch_prior_on()
    assert torch.allclose(m(p, f, sp), sp, atol=1e-7)
    with pytest.raises(ValueError):
        m(p, f)


def test_passband_mask_and_db_error():
    f = np.linspace(2.0, 6.0, 201)
    P = sample_box(3, np.random.default_rng(1))
    s = torch.as_tensor(_toy_batch(P, f))
    mask = N.passband_mask(s)
    assert mask.shape == (3, 201) and mask.any(dim=1).all()
    mean, mx = N.passband_db_error(s, s, mask)
    assert mean.item() == 0.0 and mx.item() == 0.0
    s2 = s.clone(); s2[:, 2:] *= 10 ** (-0.5 / 20)                           # -0.5 dB on S21
    mean, mx = N.passband_db_error(s2, s, mask)
    assert mean.item() == pytest.approx(0.5, abs=1e-3) and mx.item() == pytest.approx(0.5, abs=1e-3)


# ---------------------------------------------------------------------------
def test_prior_wrap_dispatch_on_toy():
    """The wrapper's calling conventions and output normalisation, using the toy function."""
    f = np.linspace(2.0, 6.0, 51)
    p = sample_box(1, np.random.default_rng(2))[0]
    s11, s21 = PW.s_prior(p, f, fn=toy_prior)
    a, b = toy_prior(p, f)
    assert np.allclose(s11, a) and np.allclose(s21, b)
    # dict / (f, p) / per-parameter signatures
    def as_dict(p, f): return {"S11": toy_prior(p, f)[0], "S21": toy_prior(p, f)[1]}
    def f_first(f, p): return toy_prior(p, f)
    def expanded(w, g1, g2, l1, l2, eps_r, h, f): return np.stack(toy_prior((w, g1, g2, l1, l2, eps_r, h), f))
    for fn in (as_dict, f_first, expanded):
        c, d = PW.s_prior(p, f, fn=fn)
        assert np.allclose(c, a) and np.allclose(d, b), fn.__name__
    batch = PW.s_prior_batch(np.stack([p, p]), f, fn=toy_prior)
    assert batch.shape == (2, 4, 51) and np.allclose(batch[0, 2], b.real)


@pytest.mark.skipif(not PW.PRIOR_PATH.exists(), reason="family prior.py not found")
def test_prior_wrap_matches_prior_py():
    """Wrapper output equals a direct call of prior.py's function, and is passive."""
    import data as D
    fn = PW.resolve_fn()
    names = D.samples_json_names()
    box = D.load_box()
    if names and box and all(n in box for n in names):
        p = np.array([0.5 * (lo + hi) for lo, hi in (box[k] for k in names)])     # box centre, data-driven length
    else:
        ds = D.load_palace_samples() if D.PALACE_DIR.exists() and any(D.PALACE_DIR.glob("*.npz")) else D.load_prior_dataset()
        p, names = ds.p[0], ds.param_names
    f = np.arange(2.0, 8.0001, 0.02)
    s11, s21 = PW.s_prior(p, f, param_names=names)
    direct = PW._as_complex_pair(PW._call(fn, p, f, names))
    assert s11.shape == f.shape and s21.shape == f.shape
    assert np.allclose(s11, direct[0]) and np.allclose(s21, direct[1])
    assert np.all(np.isfinite(s11)) and np.all(np.isfinite(s21))
    assert np.all(np.abs(s11) ** 2 + np.abs(s21) ** 2 <= 1.0 + 1e-6)


@pytest.mark.skipif(not PW.PRIOR_PATH.exists(), reason="family prior.py not found")
def test_prior_wrap_matches_prior_dataset():
    """The wrapper, called on the prior dataset's p and f, reproduces the stored S."""
    import data as D
    try:
        ds = D.load_prior_dataset()
    except FileNotFoundError:
        pytest.skip("prior dataset not written yet")
    js_names = D.samples_json_names()
    if js_names and tuple(js_names) != tuple(ds.param_names):
        pytest.skip(f"prior dataset param_names {ds.param_names} are stale vs samples.json {js_names}; awaiting regeneration")
    for i in (0, len(ds) // 2):
        s = PW.s_prior_batch(ds.p[i], ds.f, param_names=ds.param_names)[0]
        assert np.allclose(s, ds.s[i], atol=1e-4), f"sample {i}: max dev {np.abs(s - ds.s[i]).max()}"


# ---------------------------------------------------------------------------
def test_end_to_end_synthetic():
    """Stage A on the toy prior, switch, stage B on prior + known perturbation.
    The 2-seed ensemble must recover the truth to < 1 % relative L2 error on held-out designs,
    and must beat the prior-only error by a wide margin."""
    rng = np.random.default_rng(0)
    f = np.linspace(2.0, 6.0, 101).astype(np.float32)
    # stage A data: prior manifold
    Pa = sample_box(400, rng)
    Sa = _toy_batch(Pa, f)
    A = TensorSet.from_numpy(Pa, f, Sa)
    # stage B data: "FEM" truth = prior + perturbation
    Pb = sample_box(150, rng)
    Sb = np.stack([PW.pack_np(*toy_truth(p, f))[0] for p in Pb])
    Spb = _toy_batch(Pb, f)
    B = TensorSet.from_numpy(Pb, f, Sb, Spb)
    tr, va, te = np.arange(0, 110), np.arange(110, 125), np.arange(125, 150)

    preds = []
    for seed in range(2):
        torch.manual_seed(seed)
        m = N.DeepONetCorrection(n_params=Pa.shape[1], use_prior=False)
        m.p_std.fit(A.p); m.f_norm.fit(A.f)
        ra = fit(m, A.take(np.arange(360)), A.take(np.arange(360, 400)), epochs=150, lr=3e-3, batch_size=128,
                 patience=50, seed=seed, verbose=False)
        assert ra.best_val < 2e-3, f"stage A did not learn the manifold: val mse {ra.best_val}"
        m.switch_prior_on()
        assert torch.allclose(predict(m, B.take(te)), B.s_prior[te], atol=1e-6)
        fit(m, B.take(tr), B.take(va), epochs=1500, lr=2e-3, patience=300, seed=seed, verbose=False)
        preds.append(predict(m, B.take(te)))
    pred = torch.stack(preds).mean(0)
    truth = B.s[te]
    rel = (torch.linalg.norm(pred - truth) / torch.linalg.norm(truth)).item()
    rel_prior = (torch.linalg.norm(B.s_prior[te] - truth) / torch.linalg.norm(truth)).item()
    print(f"\n  synthetic e2e: rel L2 err model {100 * rel:.3f} %  prior-only {100 * rel_prior:.2f} %")
    assert rel_prior > 0.05, "perturbation too small to be a meaningful test"
    assert rel < 0.01, f"model error {100 * rel:.2f} % >= 1 %"
    pb, _ = N.passband_db_error(pred, truth, N.passband_mask(truth))
    pb0, _ = N.passband_db_error(B.s_prior[te], truth, N.passband_mask(truth))
    assert pb.item() < 0.2 * pb0.item()


def test_train_script_smoke(tmp_path, monkeypatch):
    """train.py stages A, B, baselines and eval on fake files in the expected npz formats."""
    import json
    import train as T
    import eval as E
    import data as D
    rng = np.random.default_rng(3)
    f = np.linspace(2.0, 6.0, 41)
    # prior dataset npz
    Pa = sample_box(120, rng)
    s11 = np.stack([toy_prior(p, f)[0] for p in Pa]); s21 = np.stack([toy_prior(p, f)[1] for p in Pa])
    (tmp_path / "data" / "palace").mkdir(parents=True)
    prior_npz = tmp_path / "data" / "prior_stageA.npz"
    names = np.array(["w", "g1", "g2", "l1", "l2", "eps_r", "h"])
    np.savez(prior_npz, p=Pa, f=f * 1e9, S11=s11, S21=s21, param_names=names)   # f in Hz on purpose
    # palace samples, two substrates (eps_r fixed per group)
    for i in range(40):
        p = sample_box(1, rng)[0]; p[5] = 3.5 if i < 30 else 4.2; p[6] = 0.8
        a, b = toy_truth(p, f)
        np.savez(tmp_path / "data" / "palace" / f"s{i:03d}.npz", p=p, f=f, S11=a, S21=b, wall_s=12.0 + i,
                 substrate="A" if i < 30 else "B", param_names=names)
    (tmp_path / "samples.json").write_text(json.dumps({"param_names": names.tolist(),
                                                       "box": {k: list(v) for k, v in __import__("toy_prior").BOX.items()}}))
    monkeypatch.setattr(T, "CKPT", tmp_path / "ckpt"); monkeypatch.setattr(T, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(E, "RESULTS", tmp_path / "results"); monkeypatch.setattr(E, "FIGS", tmp_path / "figs")
    monkeypatch.setattr(D, "SAMPLES_JSON", tmp_path / "samples.json")
    monkeypatch.setattr(PW, "PRIOR_FN", toy_prior)
    T.CKPT.mkdir(); T.RESULTS.mkdir(); E.FIGS.mkdir()
    argv = ["--quick", "--prior-npz", str(prior_npz), "--palace-dir", str(tmp_path / "data" / "palace"), "--quiet",
            "--c-min-group", "8", "--c-n", "2", "4"]
    T.main(["A"] + argv)
    T.main(["B"] + argv)
    T.main(["baselines"] + argv)
    T.main(["C"] + argv)
    E.main([])
    js = json.loads((T.RESULTS / "stageB.json").read_text())
    assert js["n_train"] + js["n_val"] + js["n_test"] == 30 and js["group"] == "A"
    assert json.loads((T.RESULTS / "stageA.json").read_text())["param_names"] == names.tolist()
    assert (T.RESULTS / "preds_stageB.npz").exists() and (T.RESULTS / "metrics.json").exists()
    assert json.loads((T.RESULTS / "stageC.json").read_text())["status"] == "done"
    idx = (E.FIGS / "INDEX.md").read_text()
    for fn in ("method_comparison.png", "parity_f0.png", "calibration_spread_vs_error.png", "stageC_adaptation.png", "timing.png"):
        assert (E.FIGS / fn).exists() and fn in idx
