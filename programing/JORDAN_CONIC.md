# Jordan-algebra conic solver

`//programing:jordan_conic` solves problems in the form

```
minimize    cᵀx
subject to  A x = b
            x ∈ K₁ × ... × Kₖ
```

Each block is a nonnegative orthant, Lorentz cone `{(t, u): t ≥ ||u||₂}`,
or real symmetric positive-semidefinite cone. This supports LP, SOCP, SDP,
and problems combining these cone types. Supply blocks in the same order as
coordinates of `x`, `c`, and columns of `A`.

For an SDP block of order `n`, use `n(n+1)/2` coordinates: first the `n`
diagonal entries, then upper-triangular off-diagonal entries column by column,
multiplied by `sqrt(2)`. This svec convention makes vector dot products equal
to matrix trace inner products. Constraint and objective coefficients must use
the same convention.

```cpp
#include "programing/jordan_conic.h"

// min t subject to (t, 3, 4) in the Lorentz cone; optimum 5.
Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2, 3);
A(0, 1) = 1; A(1, 2) = 1;
Eigen::VectorXd b(2), c(3);
b << 3, 4; c << 1, 0, 0;
auto result = jordan::Solve(A, b, c, {{jordan::ConeType::Lorentz, 3}});
// Inspect result.status, result.x, residuals, gap, and iterations.
```

`jordan::Solve` accepts a dense matrix; `jordan::SolveSparse` accepts a sparse
matrix without densifying when the sparse LP backend is selected. Both take the
same cone blocks, options and return type. `Options::backend` selects:

- `Backend::Auto` (default): sparse LP backend for a product of nonnegative
  cones, dense Jordan backend if any Lorentz or PSD block is present.
- `Backend::SparseLP`: sparse nonnegative-orthant backend; other cones return
  `InvalidInput` instead of silently changing the problem.
- `Backend::DenseJordan`: dense general symmetric-cone backend. Passing sparse
  input with this selection explicitly converts the matrix to dense.

`Result::backend` reports the chosen **linear algebra**, not a different
optimization algorithm. All cone types use one primal-dual Jordan-algebra
predictor-corrector iteration, with the same original-unit primal residual,
dual residual, and complementarity stopping criteria. For an orthant block the
Jordan product and Nesterov–Todd scaling reduce to elementwise operations.
The sparse backend eliminates the same scaled Newton equations into sparse
normal equations; the dense backend factors the full scaled KKT system. SOC and
PSD blocks currently require the dense factorization. The `LPSolver3` and
`SolveLP3` compatibility functions delegate to the same Jordan iteration; the
MPS command also calls the unified interface.

The shared iteration uses positive row scaling and cone-compatible column
scaling; SOC blocks use uniform scaling and PSD blocks use diagonal congruences.
It computes the Nesterov–Todd scaling point `w` satisfying `P(w)s = x`, with
`P(w)=2L(w)^2-L(w∘w)` and `L(w)z=w∘z`. With `W=P(sqrt(w))`, both scaled
variables equal `lambda=W⁻¹x=Ws`. The corrector includes the affine direction's
second-order Jordan product. PSD scaling factors use SVD, and an interior
repair is applied only for eigenvalues within eight machine epsilons of the
block scale; KKT residuals are checked on the resulting iterates in original
units. Set `JORDAN_TRACE=1` for dense-backend iteration diagnostics.

`Optimal` means the original-unit primal residual, dual residual, and gap pass
the requested relative tolerance and the backend's cone membership checks.
Iteration limit and numerical failure are reported separately; neither backend
currently certifies infeasibility or unboundedness.

The general Jordan backend uses a dense KKT factorization and is intended for
small problems. Use the sparse orthant backend for larger LPs. The solver does not yet provide
homogeneous self-dual certificates, presolve, or sparse cone factorizations.

Test with `bazel test //programing:jordan_conic_test //programing:jordan_conic_dataset_test //programing:external_conic_benchmark_test`. The external test needs `python3` on PATH.

## Analytical benchmark dataset

`//programing:jordan_conic_dataset_test` contains nine independently specified
small problems: three LPs (simplex, inequalities and coupled equalities), three
SOCPs (a norm, two independent cones and coupled cones), and three SDPs (a
fixed-diagonal 2×2 matrix, a 3×3 correlation matrix and minimum-trace
completion). Their target objectives follow from the constraints, rather than
from this solver. The test checks objectives, primal and dual residuals,
complementarity, and membership of both returned variables in their cones.
Several optima lie on the boundary.

```bash
bazel test //programing:jordan_conic_dataset_test --test_output=all
```

The pre-existing `//programing:mps_benchmark_test` covers three Netlib LP
instances through `LPSolver3`. These analytical fixtures exercise the common
Jordan solver; they are small examples, not a substitute for external large
SOCP/SDP benchmark suites or comparison against an independent solver.

The external suite also has an extended mode with separately labelled CBLIB
continuous relaxations. The benchmark results and
limitations are documented in `testdata/external/README.md`. Difficult-case
tests check that infeasible and unbounded inputs are not reported as optimal,
but the current numerical-failure result is not a certificate.

## Comparing LP backends

`//programing:jordan_lp_backend_test` runs both backends on exactly the same
six MPS instances (AFIRO, ADLITTLE, SHARE2B, SC50A, SC50B and BLEND) with
relative KKT tolerance `1e-6`. It checks objective, primal and dual residuals,
gap and nonnegativity independently for each factorization. Both passed all
six on the local Mac in the fastbuild configuration. Dense versus sparse solver
times ranged from 0.53 s versus 0.011 s on AFIRO to 17.4 s versus 0.071 s
on ADLITTLE; timings depend on machine and build configuration. A separate
one-iteration test compares the resulting primal-dual directions directly.

```bash
bazel test //programing:jordan_lp_backend_test --test_output=all
```
