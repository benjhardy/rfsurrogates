// Numerical check of web/surrogate.js against the Python side (node, no dependencies).
//
//     python core/export_web.py            # writes weights.json and check_ref.json (prior.py + torch reference values)
//     node web/check_port.mjs
//
// For the N random valid designs in check_ref.json it compares
//   (a) the JS prior with prior.py: max |dS| over the complex S11 and S21 on the grid (target < 1e-9),
//   (b) the JS ensemble mean (exact trunk) with the torch ensembles run in float64 on the same weights:
//       max |d|S21|| in dB over the grid (target < 1e-4 dB), plus |S11| and the linear |dS|,
//   (c) the fast path (tabulated trunk, cubic interpolation) against the exact JS path,
//   (d) for information, the exact JS path against torch as trained in float32 (that reference carries ~1e-6 noise in
//       S, which dB amplifies where |S| is small, so this number is not a port error).
// It also times one query on both paths and runs the inverse-design example of the page.
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { performance } from 'node:perf_hooks';

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const Surrogate = require(join(here, 'surrogate.js'));

const W = JSON.parse(readFileSync(join(here, 'weights.json'), 'utf8'));
const R = JSON.parse(readFileSync(join(here, 'check_ref.json'), 'utf8'));
const sur = Surrogate.load(W);
const f = Float64Array.from(R.f_grid_GHz);
const M = f.length, N = R.p.length;
let fail = false;
const ok = (v, lim) => (v < lim ? 'OK' : 'FAIL');
const chk = (v, lim) => { if (!(v < lim)) fail = true; };
const ex = v => v.toExponential(2);
console.log(`weights.json: format ${W.meta.format}, commit ${W.meta.commit}, ${Object.keys(W.ensembles).map(k => `${k} ${W.ensembles[k].models.length} seeds`).join(', ')}; ${N} designs x ${M} points`);

// (a) prior
let priorMax = 0;
for (let i = 0; i < N; i++) {
  const js = Surrogate.sPrior(R.p[i], f), ref = R.prior[i];
  for (let m = 0; m < M; m++) {
    priorMax = Math.max(priorMax, Math.hypot(js.s11re[m] - ref.s11re[m], js.s11im[m] - ref.s11im[m]), Math.hypot(js.s21re[m] - ref.s21re[m], js.s21im[m] - ref.s21im[m]));
  }
}
console.log(`(a) prior vs prior.py: max |dS| = ${ex(priorMax)} (target < 1e-9) ${ok(priorMax, 1e-9)}`);
chk(priorMax, 1e-9);

// (b), (d): exact JS path vs torch float64 / float32
const exactRes = {};
for (const key in R.ensembles) {
  const ref = R.ensembles[key];
  let d21 = 0, d11 = 0, d21pb = 0, dLin = 0, d21f32 = 0, d21f32pb = 0;
  exactRes[key] = [];
  for (let i = 0; i < N; i++) {
    const res = Surrogate.predict(sur, key, R.p[i], f, { exact: true });
    exactRes[key].push(res);
    let pk = -Infinity; for (let m = 0; m < M; m++) pk = Math.max(pk, ref.s21_dB[i][m]);
    for (let m = 0; m < M; m++) {
      const e21 = Math.abs(res.s21db[m] - ref.s21_dB[i][m]), e32 = Math.abs(res.s21db[m] - ref.s21_dB_f32[i][m]);
      d21 = Math.max(d21, e21); d11 = Math.max(d11, Math.abs(res.s11db[m] - ref.s11_dB[i][m])); d21f32 = Math.max(d21f32, e32);
      dLin = Math.max(dLin, Math.hypot(res.mean[2][m] - ref.s21_re[i][m], res.mean[3][m] - ref.s21_im[i][m]));
      if (ref.s21_dB[i][m] >= pk - 3) { d21pb = Math.max(d21pb, e21); d21f32pb = Math.max(d21f32pb, e32); }
    }
  }
  console.log(`(b) ${key} exact JS vs torch float64: max |d|S21|| = ${ex(d21)} dB over the grid (target < 1e-4) ${ok(d21, 1e-4)}, ${ex(d21pb)} dB in the passband; max |d|S11|| = ${ex(d11)} dB; max linear |dS21| = ${ex(dLin)}`);
  console.log(`(d) ${key} exact JS vs torch float32 as trained: max |d|S21|| = ${ex(d21f32)} dB over the grid, ${ex(d21f32pb)} dB in the passband (float32 noise, for information)`);
  chk(d21, 1e-4);
}

// (c) fast path vs exact
const buildMs = sur.prepareFast();
for (const key in R.ensembles) {
  let d21 = 0, d21pb = 0, dLin = 0;
  for (let i = 0; i < N; i++) {
    const a = exactRes[key][i], b = Surrogate.predict(sur, key, R.p[i], f);
    let pk = -Infinity; for (let m = 0; m < M; m++) pk = Math.max(pk, a.s21db[m]);
    for (let m = 0; m < M; m++) {
      const e = Math.abs(a.s21db[m] - b.s21db[m]); d21 = Math.max(d21, e); if (a.s21db[m] >= pk - 3) d21pb = Math.max(d21pb, e);
      dLin = Math.max(dLin, Math.hypot(a.mean[2][m] - b.mean[2][m], a.mean[3][m] - b.mean[3][m]));
    }
  }
  console.log(`(c) ${key} fast (tabulated trunk) vs exact JS: max |d|S21|| = ${ex(d21)} dB over the grid, ${ex(d21pb)} dB in the passband; max linear |dS21| = ${ex(dLin)}  (table build ${buildMs} ms for all models)`);
}

// timing: one query = prior + 5 seeds + metrics, as the page does on every slider move
const p0 = sur.centre.slice();
for (const [mode, exact] of [['prior', false], ['stageB', true], ['stageB', false], ['scratch', false]]) {
  for (let i = 0; i < 10; i++) Surrogate.metrics(Surrogate.predict(sur, mode, p0, null, { exact }));
  const n = exact ? 30 : 200, t0 = performance.now();
  for (let i = 0; i < n; i++) Surrogate.metrics(Surrogate.predict(sur, mode, p0, null, { exact }));
  console.log(`timing ${mode}${mode === 'prior' ? '' : exact ? ' (exact trunk)' : ' (fast path)'}: ${((performance.now() - t0) / n).toFixed(2)} ms per query (prior + network + metrics, ${M} points)`);
}

// metrics of the nominal vs the Palace reference, as a sanity check of the metric code
const refs = JSON.parse(readFileSync(join(here, 'palace_refs.json'), 'utf8')).refs;
const nom = refs.find(r => r.id === 'nominal');
const m = Surrogate.metrics(Surrogate.predict(sur, 'stageB', nom.p));
console.log(`nominal: stage-B f0 ${m.f0.toFixed(4)} GHz, BW ${(1e3 * m.bw).toFixed(0)} MHz, IL ${m.il.toFixed(2)} dB, RL ${m.rl.toFixed(1)} dB (${m.rlMethod}), split ${m.split}, spread ${m.spreadDb.toFixed(2)} dB; Palace f0 ${nom.palace.f0_centroid_GHz.toFixed(4)} GHz, BW ${(1e3 * nom.palace.bw3dB_GHz).toFixed(0)} MHz`);
const pal = Surrogate.metrics({ f, s21db: Float64Array.from(nom.s21_dB), s11db: Float64Array.from(nom.s11_dB) });
console.log(`nominal Palace curve through the JS metrics: f0 ${pal.f0.toFixed(4)} GHz, BW ${(1e3 * pal.bw).toFixed(0)} MHz, RL ${pal.rl.toFixed(1)} dB, split ${pal.split}`);

// inverse-design example of the page
const inv = Surrogate.inverse(sur, 'stageB', { f0: 5.2, bw: 0.5, rl: 10, rejections: [] });
console.log(`inverse f0 5.2 GHz / BW 0.5 GHz / RL 10 dB: ${inv.nStarts} starts (${inv.nValidStarts} valid), ${inv.nRefined} refined, ${inv.nEval} evaluations, ${inv.ms.toFixed(0)} ms`);
for (const c of inv.candidates) console.log(`  J ${c.J.toFixed(3)}  f0 ${c.m.f0.toFixed(3)}  BW ${(1e3 * c.m.bw).toFixed(0)} MHz  RL ${c.m.rl.toFixed(1)} dB  spread ${c.m.spreadDb.toFixed(2)} dB  split ${c.m.split}  p = [${c.p.slice(0, 6).map(v => v.toFixed(4)).join(', ')}]`);
for (const w of inv.warnings) console.log('  warning: ' + w);

if (fail) { console.log('CHECK FAILED'); process.exit(1); }
console.log('all checks passed');
