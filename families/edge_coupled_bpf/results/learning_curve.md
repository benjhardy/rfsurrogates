# Learning curve (held-out N_test = 26, live 26; cells: passband |S21| dB mean +- std over seeds / |f0 err| MHz / |S11| lin)

| N train | prior only | prior + DeepONet | DeepONet from scratch | plain MLP |
|---|---|---|---|---|
| 10 | 2.86 dB / 61 MHz / 0.050 | 1.83 +- 0.55 dB / 35 MHz / 0.032 (5 seeds) | 9.49 +- 5.46 dB / 185 MHz / 0.204 (5 seeds) | - |
| 25 | 2.86 dB / 61 MHz / 0.050 | 1.16 +- 0.26 dB / 26 MHz / 0.022 (5 seeds) | 2.62 +- 0.44 dB / 54 MHz / 0.078 (5 seeds) | - |
| 50 | 2.86 dB / 61 MHz / 0.050 | 0.82 +- 0.15 dB / 20 MHz / 0.017 (5 seeds) | 1.31 +- 0.36 dB / 24 MHz / 0.038 (5 seeds) | 9.98 +- 0.74 dB / 127 MHz / 0.108 (5 seeds) |
| 100 | 2.86 dB / 61 MHz / 0.050 | 0.62 +- 0.02 dB / 12 MHz / 0.013 (3 seeds) | 0.49 +- 0.04 dB / 9 MHz / 0.019 (3 seeds) | - |
| 150 | 2.86 dB / 61 MHz / 0.050 | 0.60 +- 0.01 dB / 12 MHz / 0.013 (3 seeds) | 0.40 +- 0.01 dB / 8 MHz / 0.015 (3 seeds) | - |
| 209 | 2.86 dB / 61 MHz / 0.050 | 0.54 +- 0.02 dB / 9 MHz / 0.012 (3 seeds) | 0.34 +- 0.00 dB / 5 MHz / 0.013 (3 seeds) | 2.58 +- 0.11 dB / 33 MHz / 0.045 (3 seeds) |
