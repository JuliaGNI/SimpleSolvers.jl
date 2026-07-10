# Changelog

All notable changes to SimpleSolvers.jl are documented here.

## [0.9.0]

Pre-1.0 breaking release from the 2026-07 code-review remediation. The bulk is bug
fixes to exported-yet-untested paths; the entries below are the user-visible
breaking and additive changes. (Behavioral bug fixes that do not change a public
signature are not enumerated here.)

### Removed (breaking)

- **Exported names that were undefined or dead on arrival:** `LUSolverLAPACK`,
  `BracketingMethod`, `IterativeMethod`, `algorithm`, `result`, `state`, `status`,
  `minimizer`, `gradient`, `hessian`, `print_hessian`. All were exported but either
  undefined (`UndefVarError`) or non-functional; nothing working could depend on them.
  They can be re-exported when actually implemented.
- Exported `NonlinearMethod` (removed); `NonlinearSolverMethod` is now exported in its
  place. `LinesearchMethod` is now a direct subtype of `SolverMethod`
  (was `<: NonlinearMethod`).
- Phantom type parameters: `AT` from `NonlinearSolver`; the eltype `T` from
  `NonlinearProblem` (now `NonlinearProblem{TF,TJ}`).
- The `NonlinearProblem{T}(F, [J,] n₁, n₂)` size-only convenience constructors
  (zero callers; the `{T}`=eltype syntax is invalid after dropping the phantom `T`).
- The dead `NonlinearPreconditioner` type, and the internal `_static` /
  `N_STATIC_THRESHOLD` helpers and `DEFAULT_Δ_REDUCTION` constant.
- The `pivots` field of `LUSolverCache` (its positional constructor now takes
  `A, perms, info`).
- The error-swallowing fallbacks `initialize!(x...)`, the 1-arg
  `solver_step!(::NonlinearSolver)`, and the generic two-arg `Gradient` functor;
  unsupported calls now raise a proper `MethodError`.

### Changed (breaking)

- `CurvatureCondition`'s `mode` is now a positional `Val{:Standard}()`/`Val{:Strong}()`
  argument (was a runtime `mode::Symbol` keyword) — inference-stable.
- The default `LU()` linear-solver cache for a plain matrix is now a `Matrix`, not an
  `MMatrix` (StaticArray inputs still yield an `MMatrix`; `LU(; static=true)` still
  forces one).
- `DogLegSolver(x, F, y; …)` no longer accepts a `refactorize` keyword (DogLeg
  refactorizes every step, so the option was meaningless). DogLeg now uses a carried,
  ρ-based trust-region radius (N&W Alg. 4.1; new `DogLegCache` `trust_radius[!]`
  accessors) and `solver_step!(::DogLegSolver)` no longer takes a `Δ` keyword.
- `PicardSolver` is now a residual-safeguarded fixed-point iteration and ignores any
  `linesearch` keyword (`d = −F` is not a descent direction for the `‖F‖²` merit).
- `alloc_x`/`alloc_g`/`alloc_h`/`alloc_j` reject non-floating-point element types with
  a clear error instead of a cryptic `InexactError`.

### Added

- `StrongWolfe` line search (Nocedal & Wright Alg. 3.5/3.6, bracket + zoom): the only
  line search that genuinely enforces the strong curvature condition.
- Aqua.jl, JET.jl, and construct-every-export smoke tests as CI quality gates.

### Fixed (highlights)

See `bugs.md` for the full list. Notable correctness fixes: backtracking no longer
stalls to a denormal when enforcing the curvature condition; stagnation is no longer
reported as convergence (residual-gated criteria); the DogLeg cluster (verbosity-gated
termination, wrong directional derivative, stationary-point NaN, unbounded recursion);
non-square finite-difference Jacobians; precision-aware FD step `8√eps(T)`; singular
linear systems now throw `SingularException` instead of returning NaN; and numerous
broken convenience entry points and docstrings.
