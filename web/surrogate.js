/* rfsurrogates web surrogate (edge_coupled_bpf): prior + DeepONet ensembles in plain JavaScript.
 *
 * A line-by-line port of prior.py (Pozar coupled-line prior with Kirschning-Jansen coupled microstrip,
 * Hammerstad open ends, Kobayashi dispersion) and of the forward pass of core/net.py (DeepONetCorrection:
 * standardizer, frequency normalizer, branch MLP with GELU, warp head, Fourier-feature trunk, 64 basis
 * functions, prior term on or off). Metrics follow core/eval.py (centroid f0, 3 dB band edges, split rule).
 * Works in a browser (global `Surrogate`) and in node (`module.exports`). No dependencies.
 * check_port.mjs compares this file against prior.py and the torch checkpoints.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.Surrogate = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ------------------------------------------------------------------ constants
  const C0 = 299792458.0, ETA0 = 376.730313668, PI = Math.PI;
  const PARAM_NAMES = ['w1', 'w2', 'g1', 'g2', 'l1', 'l2', 'eps_r', 'h'];
  const SUBSTRATES = { RO4350B: { eps_r: 3.66, h: 0.508, tand: 0.0037 }, FR4: { eps_r: 4.4, h: 1.6, tand: 0.02 }, alumina: { eps_r: 9.8, h: 0.635, tand: 0.0001 } };
  const C_MIN = 0.05;          // mm, design.py clearance rule
  const W50 = 1.1122;          // mm, 50 ohm microstrip on RO4350B 0.508 mm (samples.json validity.W50_mm)
  const DEAD_DB = -10.0, SPLIT_DB = 3.0;

  function tandFor(epsR) {
    let best = null, bd = Infinity;
    for (const k in SUBSTRATES) { const d = Math.abs(SUBSTRATES[k].eps_r - epsR); if (d < bd) { bd = d; best = SUBSTRATES[k]; } }
    return best.tand;
  }

  // ------------------------------------------------------------------ complex helpers ([re, im] pairs as two numbers)
  // complex sin / cos of (a + ib)
  function csin(a, b) { return [Math.sin(a) * Math.cosh(b), Math.cos(a) * Math.sinh(b)]; }
  function ccos(a, b) { return [Math.cos(a) * Math.cosh(b), -Math.sin(a) * Math.sinh(b)]; }
  function cdiv(ar, ai, br, bi) { const d = br * br + bi * bi; return [(ar * br + ai * bi) / d, (ai * br - ar * bi) / d]; }
  function cmul(ar, ai, br, bi) { return [ar * br - ai * bi, ar * bi + ai * br]; }

  // ------------------------------------------------------------------ prior.py port
  function hjA(u) { return 1 + Math.log((u ** 4 + (u / 52) ** 2) / (u ** 4 + 0.432)) / 49 + Math.log(1 + (u / 18.1) ** 3) / 18.7; }
  function hjB(er) { return 0.564 * ((er - 0.9) / (er + 3)) ** 0.053; }

  function microstrip(w, h, epsR) {
    const u = w / h;
    const fu = 6 + (2 * PI - 6) * Math.exp(-((30.666 / u) ** 0.7528));
    const z01 = ETA0 / (2 * PI) * Math.log(fu / u + Math.sqrt(1 + (2 / u) ** 2));
    const eeff = (epsR + 1) / 2 + (epsR - 1) / 2 * (1 + 10 / u) ** (-hjA(u) * hjB(epsR));
    return [z01 / Math.sqrt(eeff), eeff];
  }

  function openEndExtension(w, h, epsR) {
    const eeff = microstrip(w, h, epsR)[1], u = w / h;
    return 0.412 * h * (eeff + 0.3) / (eeff - 0.258) * (u + 0.262) / (u + 0.813);
  }

  function kobayashiEeff(fHz, eeff0, epsR, w, h) {
    const u = w / h, hM = h * 1e-3;
    const fk = C0 * Math.atan(epsR * Math.sqrt((eeff0 - 1) / (epsR - eeff0))) / (2 * PI * hM * Math.sqrt(epsR - eeff0));
    const f50 = fk / (0.75 + (0.75 - 0.332 / epsR ** 1.73) * u);
    const m0 = 1 + 1 / (1 + Math.sqrt(u)) + 0.32 / (1 + Math.sqrt(u)) ** 3;
    const mc = u <= 0.7 ? 1 + 1.4 / (1 + u) * (0.15 - 0.235 * Math.exp(-0.45 * fHz / f50)) : 1.0;
    const m = Math.min(m0 * mc, 2.32);
    return epsR - (epsR - eeff0) / (1 + (fHz / f50) ** m);
  }

  function coupledMicrostrip(w, s, h, epsR) {
    const u = w / h, g = s / h;
    const [z0, eeff] = microstrip(w, h, epsR);
    const z01 = z0 * Math.sqrt(eeff);
    const v = u * (20 + g ** 2) / (10 + g ** 2) + g * Math.exp(-g);
    const epsE = (epsR + 1) / 2 + (epsR - 1) / 2 * (1 + 10 / v) ** (-hjA(v) * hjB(epsR));
    const a0 = 0.7287 * (eeff - (epsR + 1) / 2) * (1 - Math.exp(-0.179 * u));
    const b0 = 0.747 * epsR / (0.15 + epsR);
    const c0 = b0 - (b0 - 0.207) * Math.exp(-0.414 * u);
    const d0 = 0.593 + 0.694 * Math.exp(-0.562 * u);
    const epsO = ((epsR + 1) / 2 + a0 - eeff) * Math.exp(-c0 * g ** d0) + eeff;
    const q1 = 0.8695 * u ** 0.194;
    const q2 = 1 + 0.7519 * g + 0.189 * g ** 2.31;
    const q3 = 0.1975 + (16.6 + (8.4 / g) ** 6) ** (-0.387) + Math.log(g ** 10 / (1 + (g / 3.4) ** 10)) / 241;
    const q4 = 2 * q1 / q2 / (u ** q3 * Math.exp(-g) + (2 - Math.exp(-g)) * u ** (-q3));
    const z0e = z01 / Math.sqrt(epsE) / (1 - z01 / ETA0 * q4);
    const q5 = 1.794 + 1.14 * Math.log(1 + 0.638 / (g + 0.517 * g ** 2.43));
    const q6 = 0.2305 + Math.log(g ** 10 / (1 + (g / 5.8) ** 10)) / 281.3 + Math.log(1 + 0.598 * g ** 1.154) / 5.1;
    const q7 = (10 + 190 * g ** 2) / (1 + 82.3 * g ** 3);
    const q8 = Math.exp(-6.5 - 0.95 * Math.log(g) - (g / 0.15) ** 5);
    const q9 = Math.log(q7) * (q8 + 1 / 16.5);
    const q10 = (q2 * q4 - q5 * Math.exp(Math.log(u) * q6 * u ** (-q9))) / q2;
    const z0o = z01 / Math.sqrt(epsO) / (1 - z01 / ETA0 * q10);
    return [z0e, z0o, epsE, epsO];
  }

  function filterSections(p) {
    const [w1, w2, g1, g2, l1, l2, epsR, h] = p;
    const out = [];
    for (const [w, g, l] of [[w1, g1, l1], [w2, g2, l2], [w2, g2, l2], [w1, g1, l1]]) {
      const [z0e, z0o, ee, eo] = coupledMicrostrip(w, g, h, epsR);
      out.push([w, z0e, z0o, ee, eo, l + openEndExtension(w, h, epsR)]);
    }
    return out;
  }

  // complex electrical length theta = (beta - j alpha) l for one mode; returns [re, im]
  function modeTheta(fHz, lMm, eeff, epsR, tand) {
    const beta = 2 * PI * fHz / C0 * Math.sqrt(eeff);
    const q = epsR > 1 ? epsR * (eeff - 1) / (eeff * (epsR - 1)) : 0.0;
    const alpha = beta * q * tand / 2;
    return [beta * lMm * 1e-3, -alpha * lMm * 1e-3];
  }

  /** S11, S21 of the prior for design p (mm, eps_r, h) at frequencies fGhz. Returns {s11re, s11im, s21re, s21im}. */
  function sPrior(p, fGhz, opts) {
    opts = opts || {};
    const [w1, w2, g1, g2, l1, l2, epsR, h] = p;   // eslint-disable-line no-unused-vars
    const tand = opts.tand == null ? tandFor(epsR) : opts.tand;
    const dispersion = opts.dispersion == null ? true : opts.dispersion;
    const z0 = opts.z0 == null ? 50.0 : opts.z0;
    const M = fGhz.length;
    const s11re = new Float64Array(M), s11im = new Float64Array(M), s21re = new Float64Array(M), s21im = new Float64Array(M);
    const secs = filterSections(p);
    for (let m = 0; m < M; m++) {
      const f = fGhz[m] * 1e9;
      // total ABCD = identity
      let Ar = 1, Ai = 0, Br = 0, Bi = 0, Cr = 0, Ci = 0, Dr = 1, Di = 0;
      for (const [w, z0e, z0o, ee, eo, l] of secs) {
        let eeF = ee, eoF = eo, z0eF = z0e, z0oF = z0o;
        if (dispersion) {
          eeF = kobayashiEeff(f, ee, epsR, w, h); eoF = kobayashiEeff(f, eo, epsR, w, h);
          z0eF = z0e * Math.sqrt(ee / eeF); z0oF = z0o * Math.sqrt(eo / eoF);
        }
        const the = modeTheta(f, l, eeF, epsR, tand), tho = modeTheta(f, l, eoF, epsR, tand);
        // z11 = -0.5j (z0e cot th_e + z0o cot th_o);  z21 = -0.5j (z0e csc th_e - z0o csc th_o)
        const se = csin(the[0], the[1]), ce = ccos(the[0], the[1]), so = csin(tho[0], tho[1]), co = ccos(tho[0], tho[1]);
        const cotE = cdiv(ce[0], ce[1], se[0], se[1]), cotO = cdiv(co[0], co[1], so[0], so[1]);
        const cscE = cdiv(1, 0, se[0], se[1]), cscO = cdiv(1, 0, so[0], so[1]);
        const t11r = z0eF * cotE[0] + z0oF * cotO[0], t11i = z0eF * cotE[1] + z0oF * cotO[1];
        const t21r = z0eF * cscE[0] - z0oF * cscO[0], t21i = z0eF * cscE[1] - z0oF * cscO[1];
        // multiply by -0.5j: (x + iy)(-0.5i) = 0.5y - 0.5 i x
        const z11r = 0.5 * t11i, z11i = -0.5 * t11r, z21r = 0.5 * t21i, z21i = -0.5 * t21r;
        const a = cdiv(z11r, z11i, z21r, z21i);                              // z11 / z21
        const z11sq = cmul(z11r, z11i, z11r, z11i), z21sq = cmul(z21r, z21i, z21r, z21i);
        const b = cdiv(z11sq[0] - z21sq[0], z11sq[1] - z21sq[1], z21r, z21i);  // (z11^2 - z21^2) / z21
        const c = cdiv(1, 0, z21r, z21i);                                    // 1 / z21
        const d = a;
        // total = total @ abcd
        const nAr = Ar * a[0] - Ai * a[1] + Br * c[0] - Bi * c[1], nAi = Ar * a[1] + Ai * a[0] + Br * c[1] + Bi * c[0];
        const nBr = Ar * b[0] - Ai * b[1] + Br * d[0] - Bi * d[1], nBi = Ar * b[1] + Ai * b[0] + Br * d[1] + Bi * d[0];
        const nCr = Cr * a[0] - Ci * a[1] + Dr * c[0] - Di * c[1], nCi = Cr * a[1] + Ci * a[0] + Dr * c[1] + Di * c[0];
        const nDr = Cr * b[0] - Ci * b[1] + Dr * d[0] - Di * d[1], nDi = Cr * b[1] + Ci * b[0] + Dr * d[1] + Di * d[0];
        Ar = nAr; Ai = nAi; Br = nBr; Bi = nBi; Cr = nCr; Ci = nCi; Dr = nDr; Di = nDi;
      }
      // abcd_to_s: den = a + b/z0 + c z0 + d
      const denR = Ar + Br / z0 + Cr * z0 + Dr, denI = Ai + Bi / z0 + Ci * z0 + Di;
      const n11r = Ar + Br / z0 - Cr * z0 - Dr, n11i = Ai + Bi / z0 - Ci * z0 - Di;
      const s11 = cdiv(n11r, n11i, denR, denI), s21 = cdiv(2, 0, denR, denI);
      s11re[m] = s11[0]; s11im[m] = s11[1]; s21re[m] = s21[0]; s21im[m] = s21[1];
    }
    return { s11re, s11im, s21re, s21im };
  }

  // ------------------------------------------------------------------ erf / GELU
  // erf to ~1e-14: Taylor series for |x| < 3, continued fraction for erfc beyond. Used once to fill a table;
  // the hot path is a cubic Hermite interpolation of that table (error ~ 2e-13), one exp per call.
  function erfExact(x) {
    const ax = Math.abs(x);
    if (ax < 3) {
      let term = x, sum = x, x2 = x * x;
      for (let n = 1; n < 120; n++) {
        term *= -x2 / n;
        const add = term / (2 * n + 1);
        sum += add;
        if (Math.abs(add) < 1e-17 * Math.abs(sum)) break;
      }
      return 2 / Math.sqrt(PI) * sum;
    }
    // erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + 2/(x + ...)))))
    let cf = ax;
    for (let k = 80; k >= 1; k--) cf = ax + (k / 2) / cf;
    const erfc = Math.exp(-ax * ax) / Math.sqrt(PI) / cf;
    return x > 0 ? 1 - erfc : erfc - 1;
  }
  const ERF_N = 3072, ERF_XMAX = 6.0, ERF_H = ERF_XMAX / ERF_N;
  const ERF_T = new Float64Array(ERF_N + 1), ERF_D = new Float64Array(ERF_N + 1);
  for (let i = 0; i <= ERF_N; i++) { const x = i * ERF_H; ERF_T[i] = erfExact(x); ERF_D[i] = 2 / Math.sqrt(PI) * Math.exp(-x * x) * ERF_H; }
  function erf(x) {
    const ax = x < 0 ? -x : x;
    if (ax >= ERF_XMAX) return x < 0 ? -1 : 1;
    const q = ax / ERF_H, i = q | 0, t = q - i;
    const t2 = t * t, t3 = t2 * t;
    const y = (2 * t3 - 3 * t2 + 1) * ERF_T[i] + (t3 - 2 * t2 + t) * ERF_D[i] + (-2 * t3 + 3 * t2) * ERF_T[i + 1] + (t3 - t2) * ERF_D[i + 1];
    return x < 0 ? -y : y;
  }
  const SQRT1_2 = Math.SQRT1_2;
  function gelu(x) { return 0.5 * x * (1 + erf(x * SQRT1_2)); }

  // ------------------------------------------------------------------ weights
  function b64ToF32(s) {
    let bin;
    if (typeof atob === 'function') bin = atob(s);
    else bin = Buffer.from(s, 'base64').toString('binary');
    const u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return new Float32Array(u8.buffer);
  }
  function flatten(a, out) { for (const v of a) { if (Array.isArray(v)) flatten(v, out); else out.push(v); } return out; }
  function tensorData(t) {
    if (typeof t.data === 'string') return Float64Array.from(b64ToF32(t.data));
    return Float64Array.from(flatten(t.data, []));
  }

  /** Prepare one model from weights.json for evaluation (flat Float64Array weights). */
  function prepareModel(m, nBasis, nFourier) {
    const T = n => tensorData(m.tensors[n]);
    const nIn = m.p_mean.length;
    const mod = {
      seed: m.seed, usePrior: !!m.use_prior, fLo: m.f_lo, fHi: m.f_hi,
      pMean: Float64Array.from(m.p_mean), pStd: Float64Array.from(m.p_std), fourierK: Float64Array.from(m.fourier_k),
      nIn, nBasis, nFourier, nTrunkIn: 1 + 2 * nFourier,
      bb0W: T('branch_body.0.weight'), bb0B: T('branch_body.0.bias'), bb2W: T('branch_body.2.weight'), bb2B: T('branch_body.2.bias'),
      bhW: T('branch_head.weight'), bhB: T('branch_head.bias'),
      tW: [T('trunk.0.weight'), T('trunk.2.weight'), T('trunk.4.weight'), T('trunk.6.weight')],
      tB: [T('trunk.0.bias'), T('trunk.2.bias'), T('trunk.4.bias'), T('trunk.6.bias')],
      warpW: m.tensors['warp_head.weight'] ? T('warp_head.weight') : null, warpB: m.tensors['warp_head.bias'] ? T('warp_head.bias') : null,
      bias: T('bias'),
    };
    mod.hBranch = mod.bb0B.length; mod.hTrunk = mod.tB[0].length;
    mod.scratch = { ps: new Float64Array(nIn), h1: new Float64Array(mod.hBranch), h2: new Float64Array(mod.hBranch), b: new Float64Array(4 * nBasis), w: new Float64Array(2),
      feat: new Float64Array(mod.nTrunkIn), t1: new Float64Array(mod.hTrunk), t2: new Float64Array(mod.hTrunk), t3: new Float64Array(nBasis) };
    mod.table = null; mod.tableN = 0;
    return mod;
  }

  function dense(W, b, nOut, nIn, x, y, act) {
    for (let o = 0, off = 0; o < nOut; o++, off += nIn) {
      let s = b[o];
      let i = 0;
      for (; i + 3 < nIn; i += 4) s += W[off + i] * x[i] + W[off + i + 1] * x[i + 1] + W[off + i + 2] * x[i + 2] + W[off + i + 3] * x[i + 3];
      for (; i < nIn; i++) s += W[off + i] * x[i];
      y[o] = act ? gelu(s) : s;
    }
  }

  /** Exact trunk basis t(u) (K values) for one warped normalised frequency u. */
  function trunkAt(mod, u, out) {
    const S = mod.scratch, nF = mod.nFourier, fk = mod.fourierK, feat = S.feat;
    feat[0] = u;
    for (let k = 0; k < nF; k++) { const a = u * fk[k]; feat[1 + k] = Math.sin(a); feat[1 + nF + k] = Math.cos(a); }
    dense(mod.tW[0], mod.tB[0], mod.hTrunk, mod.nTrunkIn, feat, S.t1, true);
    dense(mod.tW[1], mod.tB[1], mod.hTrunk, mod.hTrunk, S.t1, S.t2, true);
    dense(mod.tW[2], mod.tB[2], mod.hTrunk, mod.hTrunk, S.t2, S.t1, true);
    dense(mod.tW[3], mod.tB[3], mod.nBasis, mod.hTrunk, S.t1, out, true);
    return out;
  }

  // Fast path: the trunk depends on p only through the scalar u = (f_n - c(p)) exp(s(p)), so it is tabulated once
  // per model on a fine u grid and read back with 6-point Lagrange interpolation. The stage-B trunk (pretrained on
  // the prior over 2-8 GHz) has sharp features: with TABLE_STEP 1/2048 the interpolation error is ~6e-7 of the basis
  // amplitude (1/512 with a cubic stencil was 2e-3). The range covers u over the whole box (stage B -1.23..1.46,
  // scratch -0.92..1.10 at the corners); outside it the exact trunk is used, so the fast path is always defined.
  // check_port.mjs reports the fast-vs-exact difference in dB.
  const TABLE_U0 = -1.6, TABLE_U1 = 1.6, TABLE_STEP = 1 / 2048;
  function buildTable(mod) {
    const n = Math.round((TABLE_U1 - TABLE_U0) / TABLE_STEP) + 1, K = mod.nBasis, tab = new Float32Array(n * K), row = new Float64Array(K);
    for (let i = 0; i < n; i++) { trunkAt(mod, TABLE_U0 + i * TABLE_STEP, row); tab.set(row, i * K); }
    mod.table = tab; mod.tableN = n;
    return mod;
  }
  const LAG_X = [-2, -1, 0, 1, 2, 3], LAG_DEN = LAG_X.map((xa, a) => LAG_X.reduce((d, xb, b) => (b === a ? d : d * (xa - xb)), 1));
  const lagC = new Float64Array(6);
  function trunkFast(mod, u, out) {
    const K = mod.nBasis, q = (u - TABLE_U0) / TABLE_STEP, i = Math.floor(q);
    if (!(i >= 2 && i + 3 < mod.tableN)) return trunkAt(mod, u, out);
    const t = q - i, tab = mod.table;
    for (let a = 0; a < 6; a++) { let num = 1; for (let b = 0; b < 6; b++) if (b !== a) num *= (t - LAG_X[b]); lagC[a] = num / LAG_DEN[a]; }
    const c0 = lagC[0], c1 = lagC[1], c2 = lagC[2], c3 = lagC[3], c4 = lagC[4], c5 = lagC[5];
    const o0 = (i - 2) * K, o1 = (i - 1) * K, o2 = i * K, o3 = (i + 1) * K, o4 = (i + 2) * K, o5 = (i + 3) * K;
    for (let k = 0; k < K; k++) out[k] = c0 * tab[o0 + k] + c1 * tab[o1 + k] + c2 * tab[o2 + k] + c3 * tab[o3 + k] + c4 * tab[o4 + k] + c5 * tab[o5 + k];
    return out;
  }

  /** Network output for one design: real (4, M) rows S11 re, S11 im, S21 re, S21 im (correction, or full S when prior is off). */
  function correction(mod, p, fGhz, out, exact) {
    const M = fGhz.length, S = mod.scratch, K = mod.nBasis;
    const ps = S.ps;
    for (let i = 0; i < mod.nIn; i++) ps[i] = (p[i] - mod.pMean[i]) / mod.pStd[i];
    dense(mod.bb0W, mod.bb0B, mod.hBranch, mod.nIn, ps, S.h1, true);
    dense(mod.bb2W, mod.bb2B, mod.hBranch, mod.hBranch, S.h1, S.h2, true);
    dense(mod.bhW, mod.bhB, 4 * K, mod.hBranch, S.h2, S.b, false);
    let c = 0, sExp = 1;
    if (mod.warpW) { dense(mod.warpW, mod.warpB, 2, mod.hBranch, S.h2, S.w, false); c = S.w[0]; sExp = Math.exp(S.w[1]); }
    const inv = 1 / Math.sqrt(K), useTab = !exact && mod.table;
    const row0 = out[0], row1 = out[1], row2 = out[2], row3 = out[3], b = S.b, t = S.t3;
    const b0 = 0, b1 = K, b2 = 2 * K, b3 = 3 * K;
    for (let m = 0; m < M; m++) {
      const fn = 2.0 * (fGhz[m] - mod.fLo) / (mod.fHi - mod.fLo) - 1.0;
      const u = (fn - c) * sExp;
      if (useTab) trunkFast(mod, u, t); else trunkAt(mod, u, t);
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      for (let k = 0; k < K; k++) { const tk = t[k]; s0 += b[b0 + k] * tk; s1 += b[b1 + k] * tk; s2 += b[b2 + k] * tk; s3 += b[b3 + k] * tk; }
      row0[m] = s0 * inv + mod.bias[0]; row1[m] = s1 * inv + mod.bias[1]; row2[m] = s2 * inv + mod.bias[2]; row3[m] = s3 * inv + mod.bias[3];
    }
    return out;
  }

  /** Full S of one model: prior (if use_prior) + correction. `prior` may be passed in to avoid recomputing it. */
  function evalModel(mod, p, fGhz, prior, exact) {
    const M = fGhz.length, out = [new Float64Array(M), new Float64Array(M), new Float64Array(M), new Float64Array(M)];
    correction(mod, p, fGhz, out, exact);
    if (mod.usePrior) {
      const pr = prior || sPrior(p, fGhz);
      for (let m = 0; m < M; m++) { out[0][m] += pr.s11re[m]; out[1][m] += pr.s11im[m]; out[2][m] += pr.s21re[m]; out[3][m] += pr.s21im[m]; }
    }
    return out;
  }

  /** Load weights.json into ensembles: {stageB: [models], scratch: [models]} plus the grid and the box. */
  function load(W) {
    const nBasis = W.meta.n_basis, nFourier = W.meta.n_fourier;
    const ens = {};
    for (const k in W.ensembles) ens[k] = { label: W.ensembles[k].label, usePrior: W.ensembles[k].use_prior, models: W.ensembles[k].models.map(m => prepareModel(m, nBasis, nFourier)) };
    const sur = { meta: W.meta, f: Float64Array.from(W.f_grid_GHz), box: W.box, centre: W.centre, paramNames: W.param_names, substrate: W.substrate, ensembles: ens, fast: false };
    /** Tabulate every trunk (one-off, a few hundred ms); afterwards predict() uses the fast path unless {exact: true}. */
    sur.prepareFast = () => { const t0 = Date.now(); for (const k in ens) ens[k].models.forEach(buildTable); sur.fast = true; return Date.now() - t0; };
    /** Same, one model per event-loop tick (browser): progress(done, total) after each; resolves with the ms taken. */
    sur.prepareFastAsync = progress => new Promise(resolve => {
      const todo = []; for (const k in ens) for (const m of ens[k].models) todo.push(m);
      const t0 = Date.now(); let i = 0;
      const step = () => { if (i >= todo.length) { sur.fast = true; resolve(Date.now() - t0); return; } buildTable(todo[i++]); if (progress) progress(i, todo.length); setTimeout(step, 0); };
      setTimeout(step, 0);
    });
    return sur;
  }

  const db = (re, im) => 20 * Math.log10(Math.max(Math.hypot(re, im), 1e-12));

  /** Ensemble prediction. mode: 'prior' | ensemble key. Returns dB curves, per-seed dB curves, mean/std, prior dB. */
  function predict(sur, mode, p, fGhz, opts) {
    fGhz = fGhz || sur.f; opts = opts || {};
    const exact = opts.exact || !sur.fast;
    const M = fGhz.length, prior = sPrior(p, fGhz);
    const prior21 = new Float64Array(M), prior11 = new Float64Array(M);
    for (let m = 0; m < M; m++) { prior21[m] = db(prior.s21re[m], prior.s21im[m]); prior11[m] = db(prior.s11re[m], prior.s11im[m]); }
    const res = { f: fGhz, prior: { s21db: prior21, s11db: prior11, raw: prior }, mode };
    if (mode === 'prior') {
      res.s21db = prior21; res.s11db = prior11; res.std21 = new Float64Array(M); res.std11 = new Float64Array(M); res.seeds21 = [prior21]; res.seeds11 = [prior11];
      res.mean = [prior.s11re, prior.s11im, prior.s21re, prior.s21im];
      return res;
    }
    const ens = sur.ensembles[mode];
    if (!ens) throw new Error('unknown model ' + mode);
    const n = ens.models.length, mean = [new Float64Array(M), new Float64Array(M), new Float64Array(M), new Float64Array(M)];
    const seeds21 = [], seeds11 = [];
    for (const mod of ens.models) {
      const s = evalModel(mod, p, fGhz, prior, exact);
      const d21 = new Float64Array(M), d11 = new Float64Array(M);
      for (let m = 0; m < M; m++) { mean[0][m] += s[0][m] / n; mean[1][m] += s[1][m] / n; mean[2][m] += s[2][m] / n; mean[3][m] += s[3][m] / n; d21[m] = db(s[2][m], s[3][m]); d11[m] = db(s[0][m], s[1][m]); }
      seeds21.push(d21); seeds11.push(d11);
    }
    // mean of the real/imag parts (as trainer.ensemble_predict), then dB; spread = std over seeds of the dB curves
    const s21db = new Float64Array(M), s11db = new Float64Array(M), std21 = new Float64Array(M), std11 = new Float64Array(M);
    for (let m = 0; m < M; m++) {
      s21db[m] = db(mean[2][m], mean[3][m]); s11db[m] = db(mean[0][m], mean[1][m]);
      let a = 0, b = 0; for (let k = 0; k < n; k++) { a += seeds21[k][m]; b += seeds11[k][m]; } a /= n; b /= n;
      let va = 0, vb = 0; for (let k = 0; k < n; k++) { va += (seeds21[k][m] - a) ** 2; vb += (seeds11[k][m] - b) ** 2; }
      std21[m] = Math.sqrt(va / n); std11[m] = Math.sqrt(vb / n);
    }
    res.s21db = s21db; res.s11db = s11db; res.std21 = std21; res.std11 = std11; res.seeds21 = seeds21; res.seeds11 = seeds11; res.mean = mean;
    return res;
  }

  // ------------------------------------------------------------------ metrics (core/eval.py)
  function centroidFreq(f, s21re, s21im) {
    let num = 0, den = 0;
    for (let m = 0; m < f.length; m++) { const w = s21re[m] * s21re[m] + s21im[m] * s21im[m]; num += f[m] * w; den += w; }
    return num / Math.max(den, 1e-30);
  }
  function centroidFreqDb(f, s21db) {
    let num = 0, den = 0;
    for (let m = 0; m < f.length; m++) { const w = Math.pow(10, s21db[m] / 10); num += f[m] * w; den += w; }
    return num / Math.max(den, 1e-30);
  }
  function argmax(a) { let i = 0; for (let k = 1; k < a.length; k++) if (a[k] > a[i]) i = k; return i; }

  /** (f_lo, f_hi, f_peak) of the band within `level` dB of the peak, linear-interpolated (eval.band_edges). */
  function bandEdges(f, s21db, level) {
    level = level == null ? 3.0 : level;
    const i = argmax(s21db), thr = s21db[i] - level, n = f.length;
    let lo = i; while (lo > 0 && s21db[lo] >= thr) lo--;
    let hi = i; while (hi < n - 1 && s21db[hi] >= thr) hi++;
    const cross = (a, b) => { if (s21db[a] === s21db[b]) return f[a]; const t = (thr - s21db[a]) / (s21db[b] - s21db[a]); return f[a] + t * (f[b] - f[a]); };
    const fLo = s21db[lo] < thr ? cross(lo, lo + 1) : f[lo];
    const fHi = s21db[hi] < thr ? cross(hi, hi - 1) : f[hi];
    return [fLo, fHi, f[i], lo, hi];
  }

  // scipy.signal.find_peaks(x, prominence=1.0): local maxima (plateau midpoints) with prominence >= 1
  function findPeaks(x, minProm) {
    const n = x.length, peaks = [];
    let i = 1;
    while (i < n - 1) {
      if (x[i - 1] < x[i]) {
        let ahead = i + 1;
        while (ahead < n - 1 && x[ahead] === x[i]) ahead++;
        if (x[ahead] < x[i]) { peaks.push(((i + ahead - 1) / 2) | 0); i = ahead; continue; }
      }
      i++;
    }
    const out = [];
    for (const pk of peaks) {
      let j = pk, leftMin = x[pk];
      while (j >= 0 && x[j] <= x[pk]) { if (x[j] < leftMin) leftMin = x[j]; j--; }
      j = pk; let rightMin = x[pk];
      while (j < n && x[j] <= x[pk]) { if (x[j] < rightMin) rightMin = x[j]; j++; }
      if (x[pk] - Math.max(leftMin, rightMin) >= minProm) out.push(pk);
    }
    return out;
  }

  /** Two or more peaks within 6 dB of the maximum separated by a dip deeper than SPLIT_DB (eval.passband_is_split). */
  function passbandIsSplit(s21db, withinDb, dipDb) {
    withinDb = withinDb == null ? 6.0 : withinDb; dipDb = dipDb == null ? SPLIT_DB : dipDb;
    let mx = -Infinity; for (let k = 0; k < s21db.length; k++) if (s21db[k] > mx) mx = s21db[k];
    const pk = findPeaks(s21db, 1.0).filter(i => s21db[i] > mx - withinDb);
    if (pk.length < 2) return false;
    let lo = Math.min(...pk), hi = Math.max(...pk), mn = Infinity;
    for (let k = lo; k <= hi; k++) if (s21db[k] < mn) mn = s21db[k];
    return mn < mx - dipDb;
  }

  function interp(f, y, x) {
    const n = f.length;
    if (x <= f[0]) return y[0]; if (x >= f[n - 1]) return y[n - 1];
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (f[mid] <= x) lo = mid; else hi = mid; }
    const t = (x - f[lo]) / (f[hi] - f[lo]);
    return y[lo] + t * (y[hi] - y[lo]);
  }

  /** Return loss between the outer reflection zeros (prior.passband_features: up to 3 deepest |S11| minima inside
   *  f0 +- 0.6 BW). With fewer than two zeros falls back to the central half of the 3 dB band. */
  function returnLoss(f, s11db, f0, bw, fLo, fHi) {
    const band = [];
    for (let m = 0; m < f.length; m++) if (f[m] >= f0 - 0.6 * bw && f[m] <= f0 + 0.6 * bw) band.push(m);
    let zeros = [];
    if (band.length >= 5) {
      for (let j = 1; j < band.length - 1; j++) { const a = band[j]; if (s11db[a] < s11db[a - 1] && s11db[a] <= s11db[a + 1]) zeros.push(a); }
      zeros.sort((a, b) => s11db[a] - s11db[b]); zeros = zeros.slice(0, 3).sort((a, b) => a - b);
    }
    if (zeros.length >= 2) {
      let mx = -Infinity; for (let m = zeros[0]; m <= zeros[zeros.length - 1]; m++) mx = Math.max(mx, s11db[m]);
      return { rl: -mx, zeros: zeros.map(i => f[i]), method: 'between the outer |S11| zeros' };
    }
    const c = 0.5 * (fLo + fHi), hw = 0.25 * (fHi - fLo);
    let mx = -Infinity; for (let m = 0; m < f.length; m++) if (f[m] >= c - hw && f[m] <= c + hw) mx = Math.max(mx, s11db[m]);
    return { rl: isFinite(mx) ? -mx : NaN, zeros: zeros.map(i => f[i]), method: 'central half of the 3 dB band' };
  }

  /** Metrics of a prediction (from predict): f0 centroid, 3 dB BW, split flag, IL at f0, in-band RL, spread. */
  function metrics(res) {
    const f = res.f, s21 = res.s21db, s11 = res.s11db;
    const f0 = res.mean ? centroidFreq(f, res.mean[2], res.mean[3]) : centroidFreqDb(f, s21);
    const [fLo, fHi, fPeak] = bandEdges(f, s21);
    const pk = s21[argmax(s21)], thr = pk - 3.0;
    let spread = 0, cnt = 0;
    for (let m = 0; m < f.length; m++) if (s21[m] >= thr) { spread += res.std21 ? res.std21[m] : 0; cnt++; }
    const bw = fHi - fLo, rl = returnLoss(f, s11, f0, bw, fLo, fHi);
    return {
      f0, bw, fLo, fHi, fPeak, peakDb: pk, live: pk >= DEAD_DB,
      split: passbandIsSplit(s21), il: -interp(f, s21, f0), rl: rl.rl, rlMethod: rl.method, zeros: rl.zeros, spreadDb: cnt ? spread / cnt : 0,
    };
  }

  // ------------------------------------------------------------------ geometry (design.py)
  function clearance(p) {
    const [w1, w2, g1, g2] = p;
    return Math.min(g1 - Math.max(0, (w2 - w1) / 2), g2 - Math.max(0, (w1 - w2) / 2), g1 - Math.max(0, (W50 - w1) / 2));
  }
  const isValid = p => clearance(p) > C_MIN;
  const L_FEED = 3.0;          // mm, feed length de-embedded by the wave ports (build_case.L_FEED)
  /** Copper rectangles of build_case.layout(p): strips [x0, y0, dx, dy, row] (strips first), feeds, row axes, sections. */
  function layout(p) {
    const [w1, w2, g1, g2, l1, l2] = p;
    const y = [0];
    for (const dy of [w1 + g1, w2 + g2, w2 + g2, w1 + g1]) y.push(y[y.length - 1] + dy);
    const lx = 2 * l1 + 2 * l2;
    const strips = [[0, y[0] - w1 / 2, l1, w1, 0],
      [0, y[1] - w1 / 2, l1, w1, 1], [l1, y[1] - w2 / 2, l2, w2, 1],
      [l1, y[2] - w2 / 2, 2 * l2, w2, 2],
      [l1 + l2, y[3] - w2 / 2, l2, w2, 3], [l1 + 2 * l2, y[3] - w1 / 2, l1, w1, 3],
      [l1 + 2 * l2, y[4] - w1 / 2, l1, w1, 4]];
    const feeds = [[-L_FEED, y[0] - W50 / 2, L_FEED, W50], [lx, y[4] - W50 / 2, L_FEED, W50]];
    const sections = [
      { n: 1, x: [0, l1], gapY: [y[0] + w1 / 2, y[1] - w1 / 2], box: [0, y[0] - w1 / 2, l1, y[1] + w1 / 2] },
      { n: 2, x: [l1, l1 + l2], gapY: [y[1] + w2 / 2, y[2] - w2 / 2], box: [l1, y[1] - w2 / 2, l1 + l2, y[2] + w2 / 2] },
      { n: 3, x: [l1 + l2, l1 + 2 * l2], gapY: [y[2] + w2 / 2, y[3] - w2 / 2], box: [l1 + l2, y[2] - w2 / 2, l1 + 2 * l2, y[3] + w2 / 2] },
      { n: 4, x: [l1 + 2 * l2, lx], gapY: [y[3] + w1 / 2, y[4] - w1 / 2], box: [l1 + 2 * l2, y[3] - w1 / 2, lx, y[4] + w1 / 2] }];
    return { strips, feeds, rowY: y, lx, w50: W50, lFeed: L_FEED, sections };
  }
  /** Normalised distance to the nearest box edge over the varying parameters (0 = edge, 0.5 = centre). */
  function distanceToBoxEdge(p, box, names) {
    let d = 0.5;
    names.forEach((n, i) => { const [lo, hi] = box[n]; if (hi - lo <= 0) return; d = Math.min(d, Math.min(p[i] - lo, hi - p[i]) / (hi - lo)); });
    return d;
  }

  // ------------------------------------------------------------------ inverse design
  // seeded PRNG (mulberry32) so a search is repeatable
  function rng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

  // Latin hypercube in [0,1]^d
  function lhs(n, d, rand) {
    const X = [];
    for (let i = 0; i < n; i++) X.push(new Float64Array(d));
    for (let j = 0; j < d; j++) {
      const perm = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) { const k = Math.floor(rand() * (i + 1)); const t = perm[i]; perm[i] = perm[k]; perm[k] = t; }
      for (let i = 0; i < n; i++) X[i][j] = (perm[i] + rand()) / n;
    }
    return X;
  }

  /** Nelder-Mead on f: R^d -> R, from x0 with initial step h, at most maxEval evaluations. */
  function nelderMead(fn, x0, h, maxEval, tolX) {
    const d = x0.length;
    let simplex = [Float64Array.from(x0)];
    for (let i = 0; i < d; i++) { const x = Float64Array.from(x0); x[i] += h; simplex.push(x); }
    let vals = simplex.map(fn), evals = d + 1;
    const cen = new Float64Array(d);
    while (evals < maxEval) {
      const order = vals.map((v, i) => i).sort((a, b) => vals[a] - vals[b]);
      simplex = order.map(i => simplex[i]); vals = order.map(i => vals[i]);
      let size = 0;
      for (let i = 1; i <= d; i++) for (let j = 0; j < d; j++) size = Math.max(size, Math.abs(simplex[i][j] - simplex[0][j]));
      if (size < tolX) break;
      cen.fill(0);
      for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) cen[j] += simplex[i][j] / d;
      const worst = simplex[d], xr = new Float64Array(d);
      for (let j = 0; j < d; j++) xr[j] = cen[j] + (cen[j] - worst[j]);
      const fr = fn(xr); evals++;
      if (fr < vals[0]) {
        const xe = new Float64Array(d); for (let j = 0; j < d; j++) xe[j] = cen[j] + 2 * (cen[j] - worst[j]);
        const fe = fn(xe); evals++;
        if (fe < fr) { simplex[d] = xe; vals[d] = fe; } else { simplex[d] = xr; vals[d] = fr; }
      } else if (fr < vals[d - 1]) { simplex[d] = xr; vals[d] = fr; }
      else {
        const outside = fr < vals[d], xc = new Float64Array(d);
        for (let j = 0; j < d; j++) xc[j] = outside ? cen[j] + 0.5 * (xr[j] - cen[j]) : cen[j] + 0.5 * (worst[j] - cen[j]);
        const fc = fn(xc); evals++;
        if (fc < (outside ? fr : vals[d])) { simplex[d] = xc; vals[d] = fc; }
        else {   // shrink
          for (let i = 1; i <= d; i++) { for (let j = 0; j < d; j++) simplex[i][j] = simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]); vals[i] = fn(simplex[i]); evals++; }
        }
      }
    }
    const best = argminIdx(vals);
    return { x: simplex[best], f: vals[best], evals };
  }
  function argminIdx(a) { let i = 0; for (let k = 1; k < a.length; k++) if (a[k] < a[i]) i = k; return i; }

  /** Spec ranges covered by the training data; outside them the model is not trusted. */
  const TRUSTED = { f0: [4.3, 6.0], bw: [0.15, 0.8], rlMax: 15 };

  /**
   * Inverse design: spec {f0, bw, rl, rejections: [{f, db}]} (GHz, GHz, dB). Multi-start (LHS in the box, clearance
   * filtered) with the ensemble mean, then Nelder-Mead on the best starts. Objective: weighted squared f0 and BW
   * error + hinge penalties for return loss and rejection + penalties for a split passband and an invalid geometry.
   */
  function inverse(sur, mode, spec, opts) {
    opts = opts || {};
    const nStart = opts.nStart || 256, nRefine = opts.nRefine || 8, maxEval = opts.maxEval || 120, budgetMs = opts.budgetMs || 1800;
    const seed = opts.seed == null ? 1 : opts.seed, names = sur.paramNames, box = sur.box;
    const free = names.map((n, i) => i).filter(i => box[names[i]][1] > box[names[i]][0]);
    const d = free.length, now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now()), t0 = now();
    const toP = x => { const p = sur.centre.slice(); free.forEach((i, j) => { const [lo, hi] = box[names[i]]; p[i] = lo + Math.min(1, Math.max(0, x[j])) * (hi - lo); }); return p; };
    const rej = (spec.rejections || []).filter(r => isFinite(r.f) && isFinite(r.db));
    let nEval = 0;
    const evalSpec = p => {
      nEval++;
      const res = predict(sur, mode, p), m = metrics(res);
      let J = ((m.f0 - spec.f0) / 0.02) ** 2 + ((m.bw - spec.bw) / 0.05) ** 2;
      if (isFinite(spec.rl) && spec.rl > 0) J += (Math.max(0, spec.rl - m.rl) / 2) ** 2;
      for (const r of rej) J += (Math.max(0, interp(res.f, res.s21db, r.f) + r.db) / 3) ** 2;
      if (m.split) J += 10;
      if (!m.live) J += 100;
      return { J, m, res };
    };
    const objective = x => {
      let pen = 0;
      for (let j = 0; j < d; j++) { if (x[j] < 0) pen += 1e3 * x[j] * x[j]; if (x[j] > 1) pen += 1e3 * (x[j] - 1) ** 2; }
      const p = toP(x), c = clearance(p);
      if (c <= C_MIN) pen += 50 + 1e4 * (C_MIN - c) ** 2;
      return evalSpec(p).J + pen;
    };
    const rand = rng(seed), X = lhs(nStart, d, rand);
    const cands = [];
    for (const x of X) { const p = toP(x); if (!isValid(p)) continue; cands.push({ x, J: evalSpec(p).J }); }
    cands.sort((a, b) => a.J - b.J);
    const refined = [];
    for (let i = 0; i < Math.min(nRefine, cands.length); i++) {
      if (now() - t0 > budgetMs) break;
      const r = nelderMead(objective, cands[i].x, 0.08, maxEval, 1e-4);
      refined.push({ x: r.x, J: r.f });
    }
    const pool = refined.concat(cands.slice(0, nRefine)).map(c => { const p = toP(c.x); const e = evalSpec(p); return { p, J: e.J, m: e.m, valid: isValid(p) }; }).filter(c => c.valid);
    pool.sort((a, b) => a.J - b.J);
    // drop near-duplicates (same normalised point within 1 %)
    const uniq = [];
    for (const c of pool) {
      const dup = uniq.some(u => free.every(i => Math.abs(u.p[i] - c.p[i]) / (box[names[i]][1] - box[names[i]][0]) < 0.01));
      if (!dup) uniq.push(c);
    }
    const warnings = [];
    if (spec.f0 < TRUSTED.f0[0] || spec.f0 > TRUSTED.f0[1]) warnings.push(`f0 ${spec.f0.toFixed(2)} GHz is outside the trained range ${TRUSTED.f0[0]}–${TRUSTED.f0[1]} GHz`);
    if (spec.bw < TRUSTED.bw[0] || spec.bw > TRUSTED.bw[1]) warnings.push(`BW ${spec.bw.toFixed(2)} GHz is outside the trained range ${TRUSTED.bw[0]}–${TRUSTED.bw[1]} GHz`);
    if (isFinite(spec.rl) && spec.rl > TRUSTED.rlMax) warnings.push(`return loss above ${TRUSTED.rlMax} dB was not seen in training`);
    return { candidates: uniq.slice(0, 5), nStarts: X.length, nValidStarts: cands.length, nRefined: refined.length, nEval, ms: now() - t0, warnings };
  }

  return {
    PARAM_NAMES, SUBSTRATES, C_MIN, W50, DEAD_DB, SPLIT_DB, TRUSTED,
    microstrip, openEndExtension, kobayashiEeff, coupledMicrostrip, filterSections, sPrior, tandFor,
    erf, erfExact, gelu, load, prepareModel, evalModel, correction, predict, trunkAt, trunkFast, buildTable, returnLoss,
    db, centroidFreq, bandEdges, findPeaks, passbandIsSplit, interp, metrics,
    clearance, isValid, layout, distanceToBoxEdge, inverse, nelderMead, lhs, rng,
  };
});
