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
- The `minimum` export (and the corresponding `import Base.minimum`): no method for it
  was ever defined here. The optimizer subsystem it belonged to now lives in a separate
  package, and the dangling references to it (docstrings, an orphaned TikZ diagram) were
  removed.
- Phantom type parameters: `AT` from `NonlinearSolver`; the eltype `T` from
  `NonlinearProblem` (now `NonlinearProblem{TF,TJ}`).
- The `NonlinearProblem{T}(F, [J,] n₁, n₂)` size-only convenience constructors
  (zero callers; the `{T}`=eltype syntax is invalid after dropping the phantom `T`).
- The dead `NonlinearPreconditioner` type and the `DEFAULT_Δ_REDUCTION` constant.
- The error-swallowing fallbacks `initialize!(x...)`, the 1-arg
  `solver_step!(::NonlinearSolver)`, and the generic two-arg `Gradient` functor;
  unsupported calls now raise a proper `MethodError`.
- The `Options.g_restol` field (and its `g_restol(::Options)` accessor). Its role — the
  residual tolerance in the convergence check — is now filled by `f_reltol`
  (defaulting to `√eps(T)`, `g_restol`'s former value), which additionally scales with
  the initial residual. `Options(; g_restol=…)` is no longer a valid keyword.

### Changed (breaking)

- The concrete nonlinear-solver method types were renamed to drop the `Method` suffix:
  `NewtonMethod` → `Newton`, `QuasiNewtonMethod` → `QuasiNewton`
  (`QuasiNewton = Newton{false}`) and `PicardMethod` → `Picard`. The old names are gone;
  update `NonlinearSolver(NewtonMethod(), …)` call sites to `Newton()` (etc.).
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
- The diagnostics helpers `check_gradient`, `check_hessian`, `check_jacobian` and
  `print_jacobian` (and their `NewtonSolver`/`QuasiNewtonSolver` forwarders) gained an
  optional leading `io::IO` argument (defaulting to `stdout`), so their output can be
  redirected or captured. `print_jacobian` now renders via `show(io, "text/plain", J)`
  instead of `display(J)`: the terminal output is identical but is now deterministic and
  honours the given `io` (a rich frontend gets plain text rather than an HTML table).
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

Second review pass (seven further correctness fixes):
- `DogLegSolver` no longer moves the iterate to a worse point when the trust-region radius
  underflows without an acceptable step. On underflow it committed the last (rejected) trial
  as long as its merit was finite — even if that trial *increased* `‖F‖²` — violating
  monotonicity (reachable in quasi-Newton mode where a stale Jacobian's dogleg legs ascend
  the merit). It now commits the last trial only if it actually decreased the merit,
  otherwise the iterate is left unchanged (a stall the residual-gated test reports as
  non-converged). The step magnitude at underflow is `O(eps)`, so the practical effect is
  small, but the monotonicity guarantee now holds.
- `PicardSolver`'s residual-monotonicity damping loop no longer (a) re-evaluates `F` at the
  full step `α = 1` — that residual is reused from the NaN safeguard, saving one `F`
  evaluation per Picard step — nor (b) commits an *unchecked* step. The loop was bounded by
  `config.max_iterations` and, on count-exhaustion (reachable with a small
  `max_iterations`), applied an `α` shrunk one factor past the last evaluated step, so the
  committed iterate's residual was never checked. It is now bounded by the step underflow
  (`α ≤ eps`, independent of `max_iterations`) and always commits the last actually-evaluated
  iterate, preserving the `‖F(x + αd)‖ ≤ ‖F(x)‖` guarantee.
- `LinearSolver`/`LU` now **restrict the element type to floating-point** — real
  (`AbstractFloat`) or complex (`Complex{<:AbstractFloat}`). A non-float input (integer,
  rational, …) is rejected at construction with a clear `ArgumentError` instead of being
  silently promoted to `Float64`. Previously an integer matrix was promoted, but the
  promoted solver was only partially usable (`factorize!`/`ldiv!`/`solve!` were locked to
  the promoted type, so reusing it with the original integer arguments threw a `MethodError`
  or hit the `ldiv! not implemented` stub). Since the package only ever solves real/complex
  float problems, rejecting non-float input up front is clearer than promoting. ⚠ Breaking:
  `LinearSolver(LU(), A)` for an integer/rational `A` now errors; convert with `float.(A)`
  (or `complex(float.(A))`) first.
- Convergence assessment now uses the standard **`atol + rtol·‖F₀‖` residual test**. The
  successive-change criteria were gated on a fixed *absolute* residual `rfₐ ≤ g_restol`
  (`≈ √eps`), so a well-scaled solve whose residual floors at a large absolute value
  (a large-magnitude or ill-conditioned `F`, e.g. `F(x) = 10¹⁰·(x²−2)`) could never be
  reported as converged and ran to `max_iterations`. The gate is now
  `rfₐ ≤ f_abstol + f_reltol·‖F(x₀)‖`, with `atol = f_abstol` and `rtol = f_reltol`
  relative to the initial residual, so the tolerance follows the problem scale while a
  step that stalls near its initial residual is still rejected. The (previously unused)
  `Options.f_reltol` field now holds the relative tolerance and defaults to `√eps(T)`
  (was `2eps(T)`); the redundant `g_restol` field was removed (see *Removed (breaking)*).
  Note the default `f_abstol = 0` makes the default gate purely relative — there is no
  longer a fixed `√eps` absolute floor for problems whose `‖F(x₀)‖ < 1`; set `f_abstol`
  to restore one. `NonlinearSolverState` gained an `initial_residual` field (set by
  `initialize!`) to carry `‖F(x₀)‖`; the relative term is dropped (leaving the absolute
  `f_abstol` test) until the state is initialized.
- `Backtracking` no longer returns `α = 0` when the sufficient-decrease condition
  cannot be met within `max_iterations`. Returning the `α₀ = 0` anchor froze the outer
  iterate (`x .+= 0 .* d`) and spun the solve to `max_iterations`; it now returns the
  last trial step actually evaluated and stops shrinking once `α ≤ eps` instead of
  driving `α` to a denormal over all iterations.
- The custom LU solver's bare-RHS forms `solve!(x, lsolver, b)` / `solve(lsolver, b)`
  (which solve against the *stored* factorization) now throw an `ArgumentError` when the
  solver has never been factorized, instead of silently returning garbage (`ldiv!` would
  gather `b[perms[i]] = b[0]` on the zero-initialized `perms`).
- The `DogLegSolver` degrades gracefully on a singular Jacobian: with the default
  `regularization_factor = 0`, a singular factorization used to make the Newton-leg
  `ldiv!` throw a `SingularException` and abort the whole solve *before* the
  steepest-descent leg was formed. The steepest-descent (Cauchy) direction is now
  computed first and reused as the Newton leg when the factorization is singular, so the
  dogleg step degenerates to the Cauchy step — the graceful degradation the method exists
  to provide. (A plain `Newton`/`QuasiNewton` solve still throws, having no fallback.)

### Internal

- Source-file reorganization (no API or behavioral change): the solver-method type
  definitions were moved out of the standalone `src/base/methods.jl` (file removed) to
  live alongside the solvers that consume them — the `NonlinearSolverMethod` supertype
  in `nonlinear_solver.jl`, `Newton`/`QuasiNewton` (and the
  `DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER` constant) in `newton_solver.jl`, `Picard` in
  `picard_solver.jl`, and `DogLeg` in `dogleg_solver.jl`. The exported names and their
  behavior are unchanged.
- The `src/linesearch/backtracking/` subdirectory was flattened into `src/linesearch/`
  (the backtracking condition files now live alongside the other line searches). No API
  or behavioral change.
- `StrongWolfe` now composes the shared `SufficientDecreaseCondition` and
  `CurvatureCondition(…, Val(:Strong))` (issue #166) instead of re-implementing the
  Wolfe inequalities inline; the evaluation count and behavior are unchanged.
