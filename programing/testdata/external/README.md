# External LP, SOCP and SDP benchmarks

Small instances downloaded on 2026-11-11:

- `netlib/{sc50a,sc50b,blend}.mps`: decompressed from the [COIN-OR Data-Netlib mirror](https://github.com/coin-or-tools/Data-Netlib). Its LICENSE explicitly says it covers the build system, not the `.mps.gz` data; it is included for transparency, not as a claim about the data's license. Published LP objectives are available in [Hager's table](https://people.clas.ufl.edu/hager/objective-value-comparisons-for-netlib-test-problems/) and the [Netlib result summary](https://netlib.org/lp/data/).
- `cblib/sambal.cbf`: decompressed from [CBLIB](https://cblib.zib.de/download/all/sambal.cbf.gz); the CBLIB data license is in `cblib/LICENSE`. It is continuous, with free variables, Lorentz, nonnegative, and equality constraints. It is not a member of the CBLIB 2014 reference set.
- `sdplib/{truss1,control1,hinf1}.dat-s`: [SDPLIB 1.2](https://github.com/vsdp/SDPLIB), with its GPL-3.0 license in `sdplib/LICENSE`. Objectives are published in the SDPLIB README and are rounded.

Run from the repository root:

```sh
bazel build //programing:solve_mps //programing:jordan_benchmark_cli
python3 programing/external_conic_benchmark.py --verify
```

The script handles only the explicit SDPA and CBF subset present in these files;
it rejects other sections and cones. SDPA is converted to its dual and the
reported objective reverses the converted minimization sign. For CBF, equality
constraint duals are represented as a difference of nonnegative variables; the
objective is sign-reversed to report the original primal minimization. With `--verify`, the script independently recomputes residuals, gap, and cone membership, compares objectives, and exits with failure when a benchmark does not pass. `bazel test //programing:external_conic_benchmark_test` runs these checks as a regression test.
`solve_mps` reports status, primal and dual residuals, and complementarity gap
in the original standard-form units. Elapsed time for the Jordan runs includes
CLI process startup; LP timing is solver-only.

Reference calculations used Clarabel 0.11.1, NumPy 2.0.2 and SciPy 1.13.1:

```sh
python3 -m pip install clarabel==0.11.1 numpy==2.0.2 scipy==1.13.1
python3 programing/external_conic_reference.py
python3 programing/external_sambal_reference.py
```

The first compares the converted models; the second separately solves the
original CBF primal, providing an additional conversion sign check. On the
recorded run Clarabel solved all four: truss1 -8.99999627, control1 17.78462677,
hinf1 2.03259989, and sambal 3.96822045 (original primal 3.96822061). This is
an independent solver comparison, not a formal exact certificate.

Earlier local Mac results, before the predictor-corrector and PSD numerical fixes (seconds vary by machine and build):

| Instance | Solver | Objective | Reference | Outcome | Primal residual | Dual residual | Gap | Seconds |
|---|---|---:|---:|---|---:|---:|---:|---:|
| sc50a | LPSolver3 | -64.57506819 | -64.57507706 | optimal, iteration 211 | 1.37e-8 | 1.94e-6 | 2.14e-5 | 0.36 |
| sc50b | LPSolver3 | -69.99999677 | -70 | optimal, iteration 213 | 4.64e-8 | 1.95e-6 | 8.80e-6 | 0.36 |
| blend | LPSolver3 | -30.81212575 | -30.81214985 | optimal, iteration 250 | 7.28e-10 | 6.68e-6 | 6.24e-5 | 0.25 |
| truss1 | Jordan | -8.99999637 | -8.999996 | optimal, iteration 11 | 8.15e-10 | 1.15e-10 | 7.47e-8 | 0.62 |
| control1 | Jordan | 17.78462673 | 17.78463 | optimal, iteration 35 | 3.79e-9 | 8.00e-11 | 8.65e-8 | 3.26 |
| hinf1 | Jordan | 2.032599872 | 2.0326 | optimal, iteration 59 | 3.11e-14 | 1.06e-9 | 2.15e-8 | 1.92 |
| sambal | Jordan | 3.968220357 | 3.96822061 | optimal, iteration 15 | 1.73e-14 | 1.98e-8 | 2.05e-8 | 1.51 |

Jordan residuals apply to the converted models; for CBLIB those are the original
problem's dual. Failed or limited iterates are not certified feasible or optimal.

The original blend run solved an incorrectly parsed model: unnamed RHS values
were dropped. The table reflects the corrected parser and KKT checks, which use
relative tolerance 1e-6. The earlier zero objective is not a valid result for
the original instance.

The Jordan results above follow the Nesterov–Todd predictor-corrector update.
The previous unscaled method failed or reached its iteration limit on control1,
hinf1 and sambal; those results are superseded.

## Expanded suite and difficult cases (2026-11-11)

Additional Netlib `sc105`, `sc205`, `recipe` and SDPLIB `truss4`, `hinf2`,
`hinf3` files come from the same sources and licenses above. Additional CBLIB
`gbd` and `nvs03` files are **mixed-integer**. The benchmark explicitly drops
integrality and solves their continuous relaxations only; it does not solve or
claim a solution to the original mixed-integer problems. The converter refuses
integer files unless relaxation is explicitly requested. Clarabel separately
solved the original-form relaxations: `gbd` 2.2000000000 and `nvs03`
8.1521399620. Their converted duals also match these objectives.

| Problem | Solver outcome | Computed objective | Reference | Residual/gap or limit |
|---|---|---:|---:|---|
| Netlib sc105 | optimal | -52.20205188 | -52.20206121 | primal 1.21e-7; dual 1.90e-6; gap 2.16e-5 |
| Netlib sc205 | iteration limit | -52.20186938 | -52.20206121 | 400 iterations; dual 9.25e-5; gap 4.90e-4 |
| Netlib recipe | iteration limit | -266.6154452 | -266.616 | 400 iterations; dual 4.95e-4; gap 4.20e-3 |
| SDPLIB truss4 | optimal | -9.00999630 | -9.009996 | primal 1.72e-11; dual 1.56e-10; gap 1.68e-8 |
| SDPLIB hinf2 | numerical failure | 10.96705660 | ~10.96734416 | iteration 47; gap 3.15e-7 |
| SDPLIB hinf3 | numerical failure | 56.94087169 | ~56.94094238 | iteration 40; gap 1.68e-5 |
| CBLIB gbd relaxation | optimal | 2.20000000 | 2.20000000 | primal 8.07e-14; dual 8.73e-11; gap 1.25e-9 |
| CBLIB nvs03 relaxation | optimal | 8.15213995 | 8.15213996 | primal 4.96e-16; dual 3.37e-13; gap 9.02e-8 |

Netlib references are from the [Netlib index](https://netlib.org/lp/data/readme).
SDPLIB publishes rounded values for hinf2 (10.967) and hinf3 (56.9). The more
precise values above are Clarabel 0.11.1 results with status `AlmostSolved`,
not verified optimal certificates. All Jordan residuals apply to the converted
models, and CBLIB residuals apply to the continuous relaxation's dual.

The Bazel regression includes the full extended small-case suite. To run it
standalone and print all diagnostics, use:

```sh
python3 programing/external_conic_benchmark.py --extended --verify
```

The command exits nonzero if any selected case fails. Omit `--verify` to
print diagnostics without asserting convergence.

The following downloaded instances were **inspected but not solved** by the
current dense Jordan solver: CBLIB `nql30` (4,501 original variables and 6,380
constraint coordinates), CBLIB `chainsing-1000-1` (12,976 variables and
rotated cones), and DIMACS `minphase` (one PSD block of order 48, or 1,176
symmetric coordinates). The first two are from the [CBLIB 2014 continuous
set](https://github.com/HFriberg/cblib-base/blob/master/cblib/instances/set-cont-cblib2014.csv);
`minphase` and its license are from [DIMACS](https://github.com/vsdp/DIMACS).
The dense converter rejects the first before matrix allocation; the second
also needs rotated-cone support. The latter is an SDP, not an SOCP. None of
these should be counted as a passed or failed numerical solve.

`//programing:jordan_conic_test` also exercises a poorly scaled LP, a
near-singular SDP optimum, a mixed three-cone problem, dependent equalities,
and infeasible LP, SOCP and SDP examples plus an unbounded LP. In the
recorded run, the first three converge. Dependent equalities, infeasible cases,
and the unbounded case return `NumericalFailure`, rather than `Optimal`.
This is not an infeasibility or unboundedness certificate; the dependent-equality
problem is feasible and exposes the lack of rank-deficiency handling.

## Rerun after the LP and PSD scaling fixes

The sparse LP path now uses the same Jordan primal-dual predictor-corrector
iteration as the general conic solver, with sparse normal equations.
The Jordan solver uses PSD diagonal congruence equilibration, SVD-based PSD
scaling factors, roundoff-level interior repair, and validates Newton and final
KKT residuals. The full extended verification now passes. The preceding tables
are retained as historical results, and their four failure rows are superseded.

| Problem | Status | Objective | Iterations | Primal residual | Dual residual | Gap |
|---|---|---:|---:|---:|---:|---:|
| SC205 | optimal | -52.2020604598 | 31 | 9.76e-11 | 7.40e-11 | 2.35e-6 |
| RECIPE | optimal | -266.615998065 | 11 | 5.06e-8 | 1.07e-8 | 5.29e-6 |
| hinf2 | optimal | 10.96705571 | 41 | 1.68e-12 | 6.10e-11 | 7.12e-8 |
| hinf3 | optimal | 56.94077842 | 67 | 3.71e-13 | 1.41e-8 | 4.36e-7 |

LP status uses relative KKT tolerance 1e-6; Jordan uses 1e-8. Objective and
residuals are in the original standard-form units, and the external script
independently checks cone membership and KKT residuals for conic cases.
The hinf2 check compares to SDPLIB's rounded value 10.967, not Clarabel's
`AlmostSolved` 10.96734416. For hinf3 the script compares to the approximate
Clarabel `AlmostSolved` value 56.94094238 at relative tolerance 1e-5; SDPLIB's
published value is rounded to 56.9. These comparisons are not exact certificates.
