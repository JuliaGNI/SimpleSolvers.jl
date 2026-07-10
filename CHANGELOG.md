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
- The dead `NonlinearPreconditioner` type and the `DEFAULT_Δ_REDUCTION` constant.
- The error-swallowing fallbacks `initialize!(x...)`, the 1-arg
  `solver_step!(::NonlinearSolver)`, and the generic two-arg `Gradient` functor;
  unsupported calls now raise a proper `MethodError`.

### Changed (breaking)

- `CurvatureCondition`'s `mode` is now a positional `Val{:Standard}()`/`Val{:Strong}()`
  argument (was a runtime `mode::Symbol` keyword) — inference-stable.
- `DogLegSolver(x, F, y; …)` no longer accepts a `refactorize` keyword (DogLeg
  refactorizes every step, so the option was meaningless). DogLeg now uses a carried,
  ρ-based trust-region radius (N&W Alg. 4.1; new `DogLegCache` `trust_radius[!]`
  accessors) and `solver_step!(::DogLegSolver)` no longer takes a `Δ` keyword.
- `PicardSolver` is now a residual-safeguarded fixed-point iteration and no longer
  accepts a `linesearch` keyword (`d = −F` is not a descent direction for the `‖F‖²`
  merit, so a line search would be silently ignored; passing one is now an error).
- `alloc_x`/`alloc_g`/`alloc_h`/`alloc_j` reject non-floating-point element types with
  a clear error instead of a cryptic `InexactError`.
- `LinearProblem(A, y)` now stores *copies* of `A` and `y`, so the problem is usable
  right after construction. (It used to NaN-initialize both, silently discarding the
  arguments' values until an explicit `update!`.) The size-only constructors still
  allocate with `NaN`s.
- `Quadratic` and `BierlaireQuadratic` now validate their constructor parameters
  (like `Backtracking` and `StrongWolfe`); invalid values (e.g. `ε ≤ 0`,
  `s_reduction ≥ 1`) raise an `AssertionError` instead of being accepted.
- The single-step bracketing line searches (`Bisection`, `Quadratic`,
  `BierlaireQuadratic`) now fold the caller's trial step `α` into bracketing instead
  of discarding it (issue #164). In each case the `α = 0` anchor (where a descent
  direction is guaranteed decreasing) is kept as the safe fallback, and `α` is probed
  via one extra derivative evaluation:
    - `Bisection`: if `φ′(α) ≥ 0` the interval `[0, α]` is used directly as the
      bracket, otherwise `α` seeds the bracketing step scale (clamped to
      `[DEFAULT_BRACKETING_s, 1]`);
    - `Quadratic`/`BierlaireQuadratic`: bracketing starts at `α` when `φ′(α) < 0`
      (`α` on the descent side, minimiser to its right), otherwise at `0`.
  The returned step still converges to the line minimiser, but is no longer
  independent of `α`.
- Internal: the (unexported) `bracket_minimum_with_fixed_point` returns the
  bracket *with the merit values at its endpoints*, `(a, b, f(a), f(b))`; both
  quadratic line searches iterate instead of recursing and no longer re-evaluate
  the merit at points whose values are already known (2–3 fewer merit
  evaluations per interpolation iteration). The internal
  `SufficientDecreaseCondition`/`CurvatureCondition` value fields were renamed
  (`f`→`f₀`, `d`→`d₀`) so they no longer differ from the callable fields
  (`F`/`D`) only by letter case.

### Added

- `StrongWolfe` line search (Nocedal & Wright Alg. 3.5/3.6, bracket + zoom): the only
  line search that genuinely enforces the strong curvature condition.
- `solve(ls::LinearSolver, args...)`: allocating counterpart of `solve!` for a
  prebuilt `LinearSolver` (previously only `solve(::LU, …)` existed, so a
  pre-factorized solver could not be used through `solve`).
- An LU implementation of the (long-documented) `solve!(x, lsolver, b)` form that
  solves against the stored factorization; it used to unconditionally throw
  "no method implemented".
- Aqua.jl, JET.jl, and construct-every-export smoke tests as CI quality gates.
- `Options` gains `dogleg_radius_initial`, `dogleg_radius_max`, `dogleg_radius_shrink`
  and `dogleg_radius_expand` fields (defaulting to `DOGLEG_Δ_INITIAL = 1.0`,
  `DOGLEG_Δ_MAX = 1e2`, `DOGLEG_Δ_SHRINK = 0.25` and `DOGLEG_Δ_EXPAND = 2.0`), so the
  `DogLegSolver`'s trust-region radius bounds and its shrink/expand factors can be
  tuned for problems whose natural scale differs from 1 (the solver now reads these
  from `Options` instead of the hard-coded constants).

### Fixed (highlights)

See `bugs.md` for the full list. Notable correctness fixes: backtracking no longer
stalls to a denormal when enforcing the curvature condition; stagnation is no longer
reported as convergence (residual-gated criteria); the DogLeg cluster (verbosity-gated
termination, wrong directional derivative, stationary-point NaN, unbounded recursion,
trust radius now reset on solver reuse, undefined (NaN) trial merits rejected by
shrinking the radius instead of rescaling the dogleg directions); non-square
finite-difference Jacobians; precision-aware FD step `8√eps(T)`; singular linear
systems now throw `SingularException` instead of returning NaN;
`default_precision(T)` is now defined for all `AbstractFloat`s (used to error for
`Float16`/`BigFloat`); `NonlinearSolver(QuasiNewton(n), …)` now honors `n`
(the method's `refactorize` field used to be silently discarded in favor of the
default 5); `DogLegSolver(x, y; F=…)` follows the same friendly `F=missing`
pattern as the other solvers; and numerous broken convenience entry points and
docstrings.

Follow-up verification pass (independent re-review of the implemented algorithms):
the `StrongWolfe` line search no longer returns an unvalidated, freshly-doubled step
on the rare bracketing-exhaustion path (reachable with a small `max_iterations` or
`αmax = Inf`); it now returns the last trial step that satisfied sufficient decrease,
restoring its documented "never worse than Armijo" guarantee. The internal
`bracket_minimum_with_fixed_point` now stops at the *turning point* (where the merit
stops decreasing) rather than only where it climbs back above the fixed left anchor
`f(a)`, so the `Quadratic` line search no longer errors on merits whose right tail
stays below `f(a)` (e.g. one that dips to a minimum then only asymptotes back up).
The DogLeg solver no longer freezes when the carried trust-region radius underflows:
a step that enters with a collapsed radius (`Δ ≤ eps`) now resets the radius and
forces a fresh Jacobian, so it makes progress instead of silently spinning to
`max_iterations`. This is reachable in quasi-Newton mode (`refactorize > 1`), where a
stale Jacobian's steepest-descent direction need not reduce `‖F‖²`; the default
`refactorize = 1` (Jacobian refreshed every step) is unaffected.
The `Quadratic` line search's near-stationary early return now returns the bracket
point `a` at which the derivative was actually tested rather than the loop's start
`α` (they differ only when the bracketer flipped because the start was not on the
descent side). Plus minor cleanups: a dead assignment removed from `bisection`, and
the `triple_point_finder` docstring corrected to state its actual (non-strict on the
left) bracket guarantee.

### Internal

- Source-file reorganization (no API or behavioral change): the solver-method type
  definitions were moved out of the standalone `src/base/methods.jl` (file removed) to
  live alongside the solvers that consume them — the `NonlinearSolverMethod` supertype
  in `nonlinear_solver.jl`, `Newton`/`QuasiNewton` (and the
  `DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER` constant) in `newton_solver.jl`, `Picard` in
  `picard_solver.jl`, and `DogLeg` in `dogleg_solver.jl`. The exported names and their
  behavior are unchanged.
