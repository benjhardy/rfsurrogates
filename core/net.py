"""DeepONet correction model for the edge-coupled microstrip band-pass filter.

Model:  S(f; p) = S_prior(f; p) + sum_k  b_k(p) * t_k(f)

* branch net  : standardized design vector p (n_params, from the data's param_names;
                currently w1, w2, g1, g2, l1, l2, eps_r, h) -> coefficients b, shape (4, K)
                rows = (S11 re, S11 im, S21 re, S21 im), K = 64 basis functions
* trunk net   : normalized frequency f -> K basis functions t_k(f); the branch
                also outputs an affine warp (c(p), s(p)) applied to f before the
                trunk, so the basis can follow the passband as it shifts with p
* prior term  : can be switched OFF (stage A: learn the response manifold itself)
                or ON (stage B/C: learn only the gap to the Palace FEM result)

Reciprocity is built in: one S21 head, S12 := S21.  The filter is symmetric
end to end, so S22 := S11.  `to_2x2` returns the full matrix.

Passivity penalty: relu(|S11|^2 + |S21|^2 - 1), averaged over samples and
frequencies.  It is zero for any passive S.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

N_OUT = 4              # S11 re, S11 im, S21 re, S21 im
# The design-vector length is driven by the data (param_names in the npz / samples.json).
# Current data side: p = (w1, w2, g1, g2, l1, l2, eps_r, h) -> 8 inputs.  Never hard-code it.
DEFAULT_N_PARAMS = 8


def mlp(sizes: Sequence[int], act=nn.GELU, final_act: bool = False) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or final_act:
            layers.append(act())
    return nn.Sequential(*layers)


class Standardizer(nn.Module):
    """Affine map x -> (x - mean) / std with buffers, so it is saved in state_dict."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> "Standardizer":
        self.mean.copy_(x.mean(0))
        std = x.std(0, unbiased=False)
        std = torch.where(std > 1e-12, std, torch.ones_like(std))
        self.std.copy_(std)
        self.fitted.fill_(True)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class FreqNormalizer(nn.Module):
    """Map frequency (GHz) to [-1, 1] over the fitted range."""

    def __init__(self):
        super().__init__()
        self.register_buffer("f_lo", torch.tensor(0.0))
        self.register_buffer("f_hi", torch.tensor(1.0))

    @torch.no_grad()
    def fit(self, f: torch.Tensor) -> "FreqNormalizer":
        self.f_lo.fill_(float(f.min()))
        self.f_hi.fill_(float(f.max()))
        return self

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return 2.0 * (f - self.f_lo) / (self.f_hi - self.f_lo) - 1.0


class DeepONetCorrection(nn.Module):
    """S = [S_prior] + branch(p) . trunk(f).

    Parameters
    ----------
    n_basis      : K, number of basis functions (default 64)
    branch_hidden: hidden width of the branch MLP
    trunk_hidden : hidden width of the trunk MLP
    warp         : design-dependent affine warp of the trunk's frequency axis (default True)
    use_prior    : if True the forward pass adds S_prior (must be passed in);
                   if False the net itself predicts S (stage A / scratch baseline)
    """

    def __init__(self, n_params: int = DEFAULT_N_PARAMS, n_basis: int = 64, branch_hidden: int = 96,
                 branch_depth: int = 2, trunk_hidden: int = 64, trunk_depth: int = 3, n_fourier: int = 8,
                 warp: bool = True, use_prior: bool = False):
        super().__init__()
        self.n_params = n_params
        self.n_basis = n_basis
        self.n_fourier = n_fourier
        self.use_prior = use_prior
        self.p_std = Standardizer(n_params)
        self.f_norm = FreqNormalizer()
        self.branch_body = mlp([n_params] + [branch_hidden] * branch_depth, final_act=True)
        self.branch_head = nn.Linear(branch_hidden, N_OUT * n_basis)
        # trunk input: normalized f plus sin/cos Fourier features (k = 1..n_fourier) so the
        # basis can resolve narrow passbands without a very deep MLP (spectral bias fix)
        self.register_buffer("fourier_k", torch.arange(1, n_fourier + 1, dtype=torch.float32) * math.pi)
        self.trunk = mlp([1 + 2 * n_fourier] + [trunk_hidden] * trunk_depth + [n_basis], final_act=True)
        # design-dependent affine frequency warp of the trunk input, u = (f_n - c(p)) * exp(s(p)).
        # A fixed basis represents frequency-SHIFTED narrow passbands poorly (the classic DeepONet
        # translation problem); letting the branch move and stretch the frequency axis before the
        # trunk fixed that in the stage-A probe (val MSE 3.2e-3 -> 1.7e-4 on the real prior).
        # Initialised to zero = identity warp.
        self.warp = warp
        if warp:
            self.warp_head = nn.Linear(branch_hidden, 2)
            nn.init.zeros_(self.warp_head.weight)
            nn.init.zeros_(self.warp_head.bias)
        self.bias = nn.Parameter(torch.zeros(N_OUT))

    # ----- pieces -------------------------------------------------------
    def branch_features(self, p: torch.Tensor) -> torch.Tensor:
        return self.branch_body(self.p_std(p))

    def branch(self, p: torch.Tensor) -> torch.Tensor:
        """p (N,7) -> coefficients (N, 4, K)."""
        return self.branch_head(self.branch_features(p)).view(p.shape[0], N_OUT, self.n_basis)

    def warp_params(self, p: torch.Tensor) -> Optional[torch.Tensor]:
        """p (N,7) -> (N,2) = (centre shift c, log stretch s) of the frequency warp, or None."""
        if not self.warp:
            return None
        return self.warp_head(self.branch_features(p))

    def trunk_features(self, f: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        """f (M,) [, warp (N,2)] -> [u, sin(k pi u), cos(k pi u)] with shape (M, D) or (N, M, D)."""
        u = self.f_norm(f)                                      # (M,) in [-1,1]
        if w is not None:
            u = (u[None, :] - w[:, :1]) * torch.exp(w[:, 1:])   # (N,M)
        u = u.unsqueeze(-1)
        if self.n_fourier == 0:
            return u
        arg = u * self.fourier_k
        return torch.cat([u, torch.sin(arg), torch.cos(arg)], dim=-1)

    def basis(self, f: torch.Tensor, p: Optional[torch.Tensor] = None) -> torch.Tensor:
        """f (M,) -> basis (M, K); with p and warp on -> per-design basis (N, M, K)."""
        w = None if p is None else self.warp_params(p)
        return self.trunk(self.trunk_features(f, w))

    def correction(self, p: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """Real-valued network output, shape (N, 4, M)."""
        h = self.branch_features(p)
        b = self.branch_head(h).view(p.shape[0], N_OUT, self.n_basis)     # (N, 4, K)
        if self.warp:
            t = self.trunk(self.trunk_features(f, self.warp_head(h)))     # (N, M, K)
            out = torch.einsum("nok,nmk->nom", b, t)
        else:
            t = self.trunk(self.trunk_features(f))                        # (M, K)
            out = torch.einsum("nok,mk->nom", b, t)
        return out / math.sqrt(self.n_basis) + self.bias.view(1, N_OUT, 1)

    # ----- full forward -------------------------------------------------
    def forward(self, p: torch.Tensor, f: torch.Tensor,
                s_prior: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return S as a real tensor (N, 4, M): [S11 re, S11 im, S21 re, S21 im].

        s_prior: same layout, required when use_prior is True.
        """
        out = self.correction(p, f)
        if self.use_prior:
            if s_prior is None:
                raise ValueError("use_prior=True but no s_prior passed")
            out = out + s_prior
        return out

    # ----- stage switch -------------------------------------------------
    @torch.no_grad()
    def switch_prior_on(self) -> None:
        """Stage A -> stage B switch.

        After stage A the net reproduces S_prior itself.  If we simply turned the
        prior on, the output would be 2*S_prior.  So we zero the last branch
        layer and the bias: the net then outputs exactly S_prior, and stage B
        training moves the output away from the prior only where the Palace data
        asks for it.  The trunk basis and branch hidden layers (the learned
        response manifold) are kept.
        """
        self.branch_head.weight.zero_()
        self.branch_head.bias.zero_()
        self.bias.zero_()
        self.use_prior = True

    def freeze_trunk(self, freeze: bool = True) -> None:
        for q in self.trunk.parameters():
            q.requires_grad_(not freeze)

    def n_parameters(self) -> int:
        return sum(q.numel() for q in self.parameters())


class PlainMLP(nn.Module):
    """Baseline (iii): p -> S on a fixed frequency grid of M points (no prior, no trunk)."""

    def __init__(self, n_freq: int, n_params: int = DEFAULT_N_PARAMS, hidden: int = 64, depth: int = 2):
        super().__init__()
        self.n_freq = n_freq
        self.n_params = n_params
        self.p_std = Standardizer(n_params)
        self.net = mlp([n_params] + [hidden] * depth + [N_OUT * n_freq])
        self.use_prior = False

    def forward(self, p: torch.Tensor, f: torch.Tensor = None, s_prior=None) -> torch.Tensor:
        out = self.net(self.p_std(p)).view(p.shape[0], N_OUT, self.n_freq)
        return out

    def n_parameters(self) -> int:
        return sum(q.numel() for q in self.parameters())


# ---------------------------------------------------------------------------
# complex <-> real helpers
# ---------------------------------------------------------------------------
def pack(s11: torch.Tensor, s21: torch.Tensor) -> torch.Tensor:
    """Complex (N,M),(N,M) -> real (N,4,M)."""
    return torch.stack([s11.real, s11.imag, s21.real, s21.imag], dim=1)


def unpack(s: torch.Tensor):
    """Real (N,4,M) -> complex S11 (N,M), S21 (N,M)."""
    s11 = torch.complex(s[:, 0], s[:, 1])
    s21 = torch.complex(s[:, 2], s[:, 3])
    return s11, s21


def to_2x2(s: torch.Tensor) -> torch.Tensor:
    """Real (N,4,M) -> complex (N,M,2,2) with S12=S21 (reciprocity), S22=S11 (symmetry)."""
    s11, s21 = unpack(s)
    row0 = torch.stack([s11, s21], dim=-1)
    row1 = torch.stack([s21, s11], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def mag_db(z: torch.Tensor, floor: float = 1e-12) -> torch.Tensor:
    return 20.0 * torch.log10(z.abs().clamp_min(floor))


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
def passivity_penalty(s: torch.Tensor) -> torch.Tensor:
    """mean relu(|S11|^2 + |S21|^2 - 1) over all samples and frequencies."""
    power = s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2 + s[:, 3] ** 2
    return F.relu(power - 1.0).mean()


def loss_fn(pred: torch.Tensor, target: torch.Tensor, lam_passive: float = 1.0):
    mse = F.mse_loss(pred, target)
    pen = passivity_penalty(pred)
    return mse + lam_passive * pen, mse.detach(), pen.detach()


def passband_db_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """|S21| dB error inside a per-sample passband mask (N,M) -> mean, max (scalars)."""
    _, s21_p = unpack(pred)
    _, s21_t = unpack(target)
    err = (mag_db(s21_p) - mag_db(s21_t)).abs()
    m = mask.to(err.dtype)
    mean = (err * m).sum() / m.sum().clamp_min(1.0)
    mx = torch.where(mask, err, torch.zeros_like(err)).max()
    return mean, mx


def passband_mask(target: torch.Tensor, level_db: float = 3.0) -> torch.Tensor:
    """Boolean (N,M): frequencies where |S21| is within `level_db` of its per-sample peak."""
    _, s21 = unpack(target)
    db = mag_db(s21)
    peak = db.max(dim=1, keepdim=True).values
    return db >= (peak - level_db)
