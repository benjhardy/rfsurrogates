# Figures index

Held-out Palace samples: N = 26 live filters in the table below (0 dead filters with Palace peak |S21| < -10 dB excluded from the metrics; 20 have a split passband (dip > 3 dB between peaks) and are left out of the BW column only; f0 = |S21|^2 centroid; the figures still show every held-out sample)

| method | passband \|S21\| err dB (mean / max) | f0 err MHz (mean abs / max) | f0 err % | BW err MHz | \|S11\| lin err |
|---|---|---|---|---|---|
| prior only | 2.864 / 13.97 | 61.0 / 113.0 | 1.22 | 113.8 | 0.0500 |
| DeepONet from scratch | 0.320 / 1.90 | 4.2 / 12.6 | 0.08 | 26.8 | 0.0117 |
| plain MLP | 2.489 / 11.55 | 29.4 / 107.4 | 0.58 | 76.3 | 0.0372 |
| prior + DeepONet (stage B ensemble) | 0.519 / 3.97 | 7.9 / 34.0 | 0.16 | 10.8 | 0.0113 |

Calibration: Pearson r(spread, error) = 0.55; 23 % of samples have error <= 2 x spread.

Timing (ms per query): prior + DeepONet, batch 1 4.48; prior + DeepONet, batched 0.328; Palace FEM per sample 6.63e+05; scratch DeepONet, batch 1 4.2; plain MLP, batch 1 0.513

Stage C: pending (no second substrate in the Palace data: a stage-C run here would be same-substrate few-shot, not a transfer test)

| file | caption |
|---|---|
| method_comparison.png | Per-sample passband |S21| dB error, f0 error and BW error on the N=26 held-out Palace samples, for each method. Box = quartiles, triangle = mean. |
| parity_f0.png | Centre frequency (|S21|^2-weighted centroid over 3-7 GHz) predicted vs Palace, held-out samples: prior only and stage-B ensemble. |
| error_vs_box_edge.png | Per-sample passband |S21| error against the normalized distance of the design point to the nearest edge of the sampling box. |
| calibration_spread_vs_error.png | Per-sample 5-seed spread of passband |S21| (dB) against the actual ensemble error; Pearson r = 0.55. |
| examples_heldout.png | Four held-out designs (best, two middle, worst by passband error): Palace, prior, and prior + DeepONet correction, |S21| and |S11| in dB. |
| stageB_training.png | Stage B validation MSE vs epoch for the five seeds (early stopping restores the best epoch). |
| timing.png | Milliseconds per query (CPU) for the networks vs the mean Palace wall time per sample recorded in the dataset. |
