# Fidelity holdouts: medium-mesh Palace (scale 1.4) vs the coarse-mesh-trained model

10 holdout designs (h0000, h0001, h0003, h0006, h0008, h0010, h0011, h0012, h0013, h0014), stage-B ensemble of 5 seeds, centroid f0.

| comparison | N live | passband |S21| dB mean / max | |f0 err| MHz mean / max | |S11| lin |
|---|---|---|---|---|
| model vs medium | 10 | 0.91 / 3.9 | 38 / 55 | 0.027 |
| coarse vs medium | 10 | 1.03 / 3.8 | 35 / 55 | 0.025 |
| model vs coarse | 10 | 0.52 / 2.1 | 7 / 26 | 0.010 |
| prior vs medium | 10 | 1.25 / 5.0 | 18 / 34 | 0.032 |
| prior vs coarse | 10 | 2.86 / 10.1 | 52 / 89 | 0.051 |
