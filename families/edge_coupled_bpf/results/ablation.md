## 2026-10-25 04:12 live-filter held-out metrics

| variant | N live | passband |S21| dB mean / max | f0 err MHz mean / max | BW err MHz | |S11| lin err |
|---|---|---|---|---|---|
| prior only (same test set) | 7 | 2.29 / 9.3 | 71 / 87 | 34 | 0.052 |
| stage B, valid-only data (default config, 52/6/7 split) | 7 | 0.71 / 4.4 | 19 / 63 | 35 | 0.013 |
|  | | worst: s0076 2.1 dB, s0096 1.0 dB, s0135 0.6 dB | dead excluded: 0 | | |

## 2026-10-26 04:22 live-filter held-out metrics

| variant | N live | passband |S21| dB mean / max | f0 err MHz mean / max | BW err MHz | |S11| lin err |
|---|---|---|---|---|---|
| prior only (same test set) | 26 | 2.86 / 14.0 | 61 / 113 | 114 | 0.050 |
| stage B, matched schedule (3 seeds, 6000 epochs, no early stop) | 26 | 0.50 / 5.0 | 8 / 29 | 10 | 0.011 |
|  | | worst: s0024 1.9 dB, s0523 1.1 dB, s0245 0.9 dB | dead excluded: 0 | | |

## 2026-10-26 04:22 live-filter held-out metrics

| variant | N live | passband |S21| dB mean / max | f0 err MHz mean / max | BW err MHz | |S11| lin err |
|---|---|---|---|---|---|
| prior only (same test set) | 26 | 2.86 / 14.0 | 61 / 113 | 114 | 0.050 |
| stage B, default schedule (5 seeds, patience 300) | 26 | 0.52 / 4.0 | 8 / 34 | 11 | 0.011 |
|  | | worst: s0024 1.7 dB, s0523 1.1 dB, s0245 0.9 dB | dead excluded: 0 | | |

