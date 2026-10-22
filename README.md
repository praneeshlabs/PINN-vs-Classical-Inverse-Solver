# Inverse PINN versus classical least squares for drug transport in tissue

## Question

For a 1D reaction diffusion slab with two unknowns, the reaction to diffusion ratio phi2 and the surface
mass transfer number Bi, where does an inverse PINN recover the parameters better than a classical
least squares solver, and where is neither method able to, because the data cannot identify the parameters?

## Experiments

| Step | What varies | Question it answers |
|---|---|---|
| tune | PINN size, learning rate, physics weight | Is the PINN a straw man? Settings are chosen on separate data by held out RMSE only |
| bench | 3 regimes, noise 0, 2, 5 percent, 5000 fitting points | Head to head accuracy, cost, and distance from the best possible error |
| sparse | 50 to 5000 fitting points | Does the PINN help when data are scarce? |
| map | 5 by 5 grid of phi2 and Bi at 5 percent noise and 500 points | Where in parameter space does each method win, and where is the problem unidentifiable? |
| mismatch | data from saturable clearance, both methods fit the linear model | Which method degrades less when the model is wrong? |
| arch | 2x16, 2x16 with 3x epochs, 2x64, 4x32, 4x64 | Is the small network limited by capacity or by training budget? |

## How to run

```
python check.py                          solver, bound, classical and PINN checks, ends with PASS or FAIL
python tune.py --workers 4               choose PINN settings
python run.py bench --workers 4          also: sparse, map, mismatch, arch, demo
python report.py                         tables and figures in results/
```

Add `--smoke` to any command for a tiny fast run that writes to results_smoke, to test the pipeline.
`--workers N` runs N fits in parallel. Finished jobs are saved and skipped on a rerun.

## PINN design

- The network input is s = sqrt(t / T). The solution changes like sqrt(t) near the surface at early times, so in s the layer is smooth and the observation grid is evenly spaced.
- The PDE is written in s as u_s = 2 T s (u_xx - phi2 u). The initial condition is exact by construction.
- Training is staged. Data only for the first fifth of the epochs, then the physics terms ramp in. Without this the parameters collapse to a trivial solution before the field is fitted.
- The data term is weighted by 1000 to 10000, in line with the noise precision. tune.py searches this weight.
- Fresh collocation points every Adam step, then L-BFGS on a fixed set.

## Fairness rules

- Same data, same split, same three starting guesses for both methods.
- Best of three is picked by held out RMSE only. Runtime counts all three fits.
- Data come from a 240 cell mesh, the classical solver uses 80 cells.
- Errors are judged against the Cramer Rao bound, the smallest error any unbiased fit can reach.
- Tuning data use seeds that never appear in the benchmark.

## Hypotheses, written before any result exists

1. Dense data, correct model (bench): classical lands close to the bound and is far cheaper. The PINN does not beat it.
2. Sparse data (sparse): the PINN gap to classical narrows as data shrink, but a win needs the cells where the bound is also loose.
3. Wrong model (mismatch): the PINN field error degrades less than classical, because its physics term is soft.
4. Regime map (map): both methods fail in the same cells, the ones with a large best possible error. Failure there is identifiability, not the method.
5. Capacity (arch): 2x16 underfits the reaction regime, and part of the gap closes with more epochs rather than more weights.

A hypothesis that turns out false is still a result. Report it.

## Metrics reported per condition

Median error with quartiles and a bootstrap interval, error divided by the best possible error,
share of runs within twice the best possible error, signed bias of each parameter, paired log10 error
ratio with interval, share of datasets where the PINN wins, held out RMSE over noise sigma (1 means
the noise floor), field error against the clean truth, spread between the three starts, valid fit rate,
runtime, and the correlation between the two parameter estimates implied by the data.

## Reading the figures

- `*_scoreboard.png`: log10 of PINN error over classical error with a 95 percent interval. Below zero, the PINN wins.
- `*_parameter_error.png`: median and quartiles per parameter, dashed line is the best possible error.
- `*_field_error.png`: error of the recovered concentration field against the clean truth.
- `*_cost.png`: runtime against error.
- `regime_map.png`: who wins in each cell, each method's error, and the best possible error.
- `arch_study.png`: error, held out RMSE over noise sigma (1 means the network reaches the noise floor), time.
- `profiles.png`: concentration profiles of truth, noisy data and both fits.

## Known limits, state them in the report

- Synthetic data only. No claim about real tissue.
- One PDE family, Gaussian noise with a constant scale relative to the peak.
- The PINN search is 12 random settings on three datasets. A larger search could change the picture.
- Validation points are random grid points, so they are interpolation points.
- Under mismatch there is no true parameter. Compare field error there, and read parameter error as bias.
- The map uses one noise level and one data size.
- Zero noise repeats differ only by the split and the PINN initialization.
- Wall clock depends on the implementation, so cost ratios are for this code on this machine.
