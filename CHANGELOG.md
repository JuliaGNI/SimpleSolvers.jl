# Changelog

All notable changes to SimpleSolvers.jl are documented here.

## [0.12.0]

Two defects reported from downstream by
[GeometricOptimizers.jl](https://github.com/JuliaGNI/GeometricOptimizers.jl), filed there as **D3**
and **D4**. **Breaking**: a `NonlinearSolver` no longer emits line-search warnings from inside its
iteration, and `NonlinearSolverStatus` gains a field.

### `Bisection` no longer reports success when it cannot bracket

#### The gap

`_bisection_core` returned `(α, converged::Bool, n)` and set `converged = true` on the branch that
handles *no sign change between the endpoints*:

```julia
if y₀ * y₁ > zero(y₀)                                  # no sign change
    report_bisection_nobracket(α₀, α₁, y₀, y₁, config)
    return (abs(y₀) ≤ abs(y₁) ? α₀ : α₁), true, n      # ← converged = true
end
```

The intent was that a flat derivative at a minimum is benign. The effect was that the core could
not tell *"found the root"* from *"gave up"*, and the endpoint of a bracket that never existed was
handed on as if it were the located line minimiser.

That claim then propagated. `Bisection` bisects ``\varphi'`` but brackets on ``\varphi``, so the
two disagree exactly when something is wrong — a stale or regularized Jacobian, an inexact linear
solve, a non-smooth merit. In that case the returned endpoint typically does not improve the merit
either, and the search classified it as `LINESEARCH_FLOOR`: *"no line search can make progress
here"*. The outer iteration acts on that (`flag_stall!`, then `max_stalls`), so a failure to
bracket was being read as a property of the problem. Downstream it contributed to a diverging
`_BFGS` + `Bisection` + `Geodesic` run in which the step was taken regardless.

#### Changed

- `_bisection_core` now returns a `BisectionOutcome` — `BISECTION_CONVERGED`,
  `BISECTION_NOBRACKET` or `BISECTION_EXHAUSTED` — in place of the `Bool`. Internal, like the
  `Symbol` that `_triple_point_core` returns. The `α` it returns is unchanged in every case.
- `bisection` dispatches on it, so a spent budget and a failed bracket get the wording that fits
  each (*"I did not narrow the interval far enough"* versus *"there is no root in it at all"*;
  only the first is fixed by a larger budget).
- The `Bisection` **line search** classifies a failed bracket by the merit alone, with
  `LINESEARCH_FLOOR` disallowed: `LINESEARCH_DECREASED` when the returned step still beats
  ``\varphi(0)`` by more than ``\tau``, `LINESEARCH_EXHAUSTED` when it does not. `LINESEARCH_FLOOR`
  is now reachable only from a bisection that actually converged. The retry from the ``\alpha = 0``
  anchor carries its own verdict through for the same reason; it used to be discarded.
- The standalone `bisection`'s no-bracket warning moved from `verbosity ≥ 2` to `≥ 1`. It sat at 2
  only because of the benign line-search case, and the line search no longer routes through
  `bisection` — it calls `_bisection_core` and reports through its `LinesearchStatus`. For a
  caller asking for a root, "there is no root in your interval" is a genuine failure.

### A line search reports to its caller, not to the user

#### The gap

The `maxlog` caps on the line-search warnings were keyed on the *source location* of the `@warn`,
so they were process-global and were never reset between `solve!` calls. Once spent, a message was
gone for the rest of the session, including for later solves of entirely different problems — a
genuine line-search failure in the tenth solve of a run was silent. This mattered because the log
was the only channel by which a rejected line search was visible at all.

The caps were a symptom. The channel split already existed — `solve_with_status` returns a
`LinesearchStatus` and emits nothing, `solve` adds `linesearch_warnings` — but `solver_step!`
called `solve_with_status` and then `linesearch_warnings` *anyway*, once per outer iteration. A
solve that cannot make progress asks the line search for an impossible decrease at every one of its
iterations, so that call is what turned one diagnosis into thousands of identical messages, and the
caps existed to hold it back.

#### Changed

- `solver_step!` no longer calls `linesearch_warnings`. A line search now reports to its *caller*
  through `LinesearchStatus`, and to the *user* only when the user called `solve(ls, α)` directly.
  One direct call yields one message, so there is nothing left to rate limit and the three
  `maxlog` caps in `report_linesearch_status` are **removed** rather than reimplemented.
- One deliberate loss: a solver at `verbosity ≥ 2` no longer gets `curvature_diagnostic` per
  iteration, which cost a full Jacobian evaluation each time.

#### Added

- `NonlinearSolverState` tallies the `LinesearchOutcome` of every step (`record_linesearch!`),
  reset per solve by `initialize!` exactly like `stalls`.
- `NonlinearSolverStatus` carries a snapshot of that tally, read with `linesearch_outcomes`,
  `linesearch_failures` and `dominant_linesearch_outcome`. This is the programmatic channel the
  package lacked: a caller that wants to *act* on a rejected line search — restart an approximate
  Hessian, fall back to steepest descent — reads the status instead of scraping the log. Not
  exported, per the convention the other status predicates follow.
- The stagnation and no-progress messages of `nonlinear_solver_warnings` name what the line search
  reported and how often ("the line search reported `LINESEARCH_NO_DESCENT` on 194 of the 200
  step(s)"), so a failed solve names the cause and not only the symptom. A solve that converged
  despite line-search failures says the same at `verbosity ≥ 2`, as an `@info`.

### The solver's own report backs off instead of going silent

`nonlinear_solver_warnings` fires once per `solve!` and is the one place a cross-solve cap is still
needed — a time-stepping loop over an unattainable tolerance would otherwise report at every step.
Its three repeatable messages are now gated on `should_report!`, which reports the 1st, 2nd, 4th,
8th … occurrence of a diagnosis rather than the first three and then nothing ever again. Keys
include the dominant line-search outcome, so a solve that starts failing for a *new* reason is
reported at once.

Be clear about the trade: a *repeating* diagnosis is reported on occurrences 1, 2, 4, 8, 16 …, so
occurrence 10 is silent and occurrence 16 is not. Every occurrence is not promised — that would be
the flood back. `verbosity = 0` still silences a solver completely, and the status is still the way
to act on an outcome rather than read about it. `reset_warning_counts!` clears the counters.

## [0.11.0]

Four independent changes. **Breaking**: `Backtracking` loses its `α₀` key and its positional
constructor, and gains an opt-in expansion phase. Additive: a `NonlinearProblem` can be solved
without assembling a solver by hand, and a `NonlinearSolver` reports — and, on request, stops —
a solve that is *not progressing*, the case
[issue #173](https://github.com/JuliaGNI/SimpleSolvers.jl/issues/173) found next to the
`max_stalls` machinery of 0.10.0. Fixed: the nonlinear solvers looked for `NaN` where they
should have looked for any non-finite value, so an *overflowed* residual passed every guard
they have —
[issue #130](https://github.com/JuliaGNI/SimpleSolvers.jl/issues/130) — and could be reported
as convergence.

### Convenience entry points for solving a `NonlinearProblem`

[Issue #159](https://github.com/JuliaGNI/SimpleSolvers.jl/issues/159).

#### The gap

`NonlinearProblem` was exported but was not usable as an argument anywhere: no solver constructor
accepted one, so a hand-built problem forced the seven-argument low-level constructor, with the
linear problem, the linear solver, the line search and the cache all supplied by the caller. The
`NewtonSolver` docstring had advertised the missing capability — *"can be called with a
`NonlinearProblem` or with a `Callable`"* — since before it existed. Nor was there a `solve`/`solve!`
taking a problem and a method, which both of the other subsystems have: `solve(lu, ls)` on the
linear side, `solve(prob, method, α, params, config)` on the line-search side.

#### Added

- `NewtonSolver(x, nlp::NonlinearProblem, y = zero(x))` and the same form for `PicardSolver` and
  `DogLegSolver`, hence also `NonlinearSolver(method, x, nlp)`. The residual prototype `y` only
  supplies a size and a type — nothing computed from it survives — so it defaults to `zero(x)`,
  which is what a square system needs, and every system here is square because the Jacobian gets
  factorized. (`zero` and not `similar`: `alloc_j` broadcasts over `y`, so an uninitialized
  prototype throws an `UndefRefError` for an element type whose `similar` leaves undefined
  references, such as `BigFloat`.) A Jacobian stored in the problem takes precedence over autodiff,
  exactly as the `DF!` keyword does.
- `solve!(x, prob, method, args...; kwargs...)`, which builds the solver and overwrites `x` with the
  solution, and `solve(x₀, prob, method, args...; kwargs...)`, which returns the solution as a new
  array and leaves `x₀` alone — `solve!`'s signature minus the bang. The trailing positional
  arguments go through to the solver-level `solve!`, so they are the problem's `params`, optionally
  preceded by a `NonlinearSolverState`; the keywords are the constructor's plus `Options`'.
- `solve_with_status!(x, s)` and `solve_with_status!(x, prob, method, params; kwargs...)`, returning
  the `NonlinearSolverStatus` instead of the solution. A wrapper discards the solver it built, so
  `status(s, state)` — which needs both — is otherwise out of reach; this builds the state itself.
  The `!` is honest here, unlike in the line search's `solve_with_status`, whose `α` is a number.

Each wrapper call constructs a solver (a Jacobian with its ForwardDiff configuration, a
factorization cache, the line-search buffers), so this is the convenience path and not the one for a
loop: that should still build one `NonlinearSolver` and reuse it. The docstrings say so.

#### Changed

The three `(x, F, y)` constructors are now one-line delegations to their `NonlinearProblem`
counterparts — the assembly recipe they each carried a copy of exists once per solver now. No
behaviour change: `NonlinearProblem(F, DF!, x, y)` stores `J = DF!`, so the resulting
`resolve_jacobian` call is the one they made before.

### `Backtracking` can lengthen a step

`Backtracking` can now lengthen a step, and no longer carries a field that pretended to
configure one. Both halves of [issue #174](https://github.com/JuliaGNI/SimpleSolvers.jl/issues/174).

#### The defects

**`Backtracking.α₀` was never read.** It was stored, documented as *"the initial step size α"*,
printed by `show`, compared by `isapprox` and converted by `change_precision` — but neither
`solve` nor `solve_with_status` ever looked at it. The trial step was, and remains, the `α`
argument. `Backtracking(; α₀ = 10.0)` therefore configured nothing.

**The search could only shrink.** A backtracking search returns the trial step it was given
whenever that step is acceptable, so on a direction whose natural scale is *larger* than the
trial step it pins `α` at the caller's ceiling on every iteration. On the SVD test problem of
JuliaGNI/GeometricOptimizers.jl#31 that cost a DFP direction — which wants `α ≈ 11` throughout —
**49 679** iterations against **134** for `Bisection`, whose rightward bracketing is immune. It
is a property of the search, not of DFP: BFGS, already scaled like a Newton step, is unaffected
(113 against 143). Handing the same `Backtracking` a trial step of 3 instead of 1 is worth a
factor of 217, which is precisely the knob `α₀` appeared to offer and did not.

#### Fixed

- **Breaking**: the `α₀` key of `Backtracking` and the constant `SimpleSolvers.DEFAULT_ARMIJO_α₀`
  are removed, along with the positional `Backtracking{T}(α₀, c₁, c₂, p[, τ_ulps])` form. The
  trial step now has exactly one source, the `α` argument of `solve`/`solve_with_status`. No
  behaviour changes, because nothing read the field.
- `Backtracking` gained an **expansion phase**, behind a new `expand` key that is `false` by
  default: with it set, an accepted *first* trial step is lengthened while each longer trial
  still satisfies the sufficient decrease condition and strictly improves the merit. A shrunken
  step is never expanded again — the longer steps below it have already been rejected. Two
  further keys tune it, `q` (an upper bound on the growth factor per round, the counterpart of
  `p`) and `nexpand` (the cap on expansion trials, applied from *within* the
  `linesearch_max_iterations` of `Options` rather than beside it, so the whole search still
  spends at most that many merit evaluations). This is the one place where a line search leaves
  the interval `[0, α]` the caller offered: the largest step it can try is `q^nexpand · α`, a
  thousand times the trial step on the defaults. A trial whose merit is not finite is rejected at
  the cost of that one evaluation, but a merit that *throws* outside its domain is the caller's
  to guard — one more reason the phase is opt-in.
- The step it grows to is chosen by a new `SimpleSolvers.backtracking_extrapolation`, from the
  *same* quadratic model through `φ(0)`, `φ'(0)` and `φ(α)` that `backtracking_interpolation`
  uses on the way down. All three values are already known when the trial step is accepted, so
  the decision whether to expand at all costs no merit evaluation: a direction scaled like a
  Newton step is at its model minimum and returns after the single trial the shrink-only search
  would have made. That is what allowed the phase to be worth having at all — for the merit of a
  `NonlinearSolver`, one evaluation is a full residual evaluation. It is also why the phase does
  not consult the curvature condition, which would cost a full Jacobian per trial; `StrongWolfe`
  remains the method for that.

#### Measured

On the `St(20,3)²` SVD problem of GeometricOptimizers (`test/optimizer_convergence/svd_optim.jl`,
seed 1234), iterations to convergence:

| method + retraction | `Backtracking` | `Backtracking(; expand = true)` | `Bisection` |
|---|---|---|---|
| `_DFP` + Geodesic | 49 679 | **830** | 134 |
| `_DFP` + Cayley | 29 081 | **1 237** | 96 |
| `_BFGS` + Geodesic | 113 | 93 | 143 |
| `_BFGS` + Cayley | 136 | 118 | 93 |

The `Backtracking` column reproduces 0.10.1's numbers exactly, which is the check that the
default path is untouched. The well-scaled `_BFGS` rows are not merely unharmed but slightly
better, because an occasional step *is* longer than the trial one. `_DFP` is still behind
`Bisection` on iteration count, but ahead of it on wall clock (0.2 s against 0.5 s for
Geodesic): a `Bisection` iteration spends of the order of 580 merit evaluations against
`Backtracking`'s 25.

On `nexpand`, the same problem: `1` is too few for `_DFP` (it does not converge within 60 000
iterations), `2` and above all do, hence the default of 3. On `q`, the range 4–100 is a plateau
rather than a cliff — `q = 100` reaches 594/398 on the two `_DFP` rows — and the default of 10 is
the conservative point on it rather than the best on this one problem.

**Behaviour is unchanged unless `expand = true`.** No expectation of the 0.10.1 test suite was
adjusted to make this pass — every iteration count, `trials` count and zero-allocation assertion
stands as it was, and the only edits to the test files are new cases and the addition of
`Backtracking(T; expand = true)` to loops that already ran over every method. `NewtonSolver`'s
default line search still shrinks only. GeometricIntegrators' Runge-Kutta suite
passes (128 assertions) with bit-identical trajectories for `Gauss(1)`…`Gauss(4)`. To fix an
under-scaled direction, ask for it:
`NewtonSolver(x, F, y; linesearch = Backtracking(T; expand = true))`.

### Solves that get nowhere

`max_stalls` (0.10.0) catches a solve whose **iterate has frozen**: the step has fallen below the
round-off level of `x` while the residual is not small, so no progress is possible along the
current direction. Issue #173 reports the case next to it — the iterate keeps moving perfectly
normally while the residual sits on a floor far above the requested tolerance. No existing
criterion can fire, by construction: `iterate_settled` is false because the step is not small,
`residual_small` is false because the residual is not either, and `stalled_step` needs both. Such
a solve spends `max_iterations` in full and reports `"Solver took 1000 iterations."`, which names
a symptom and no cause.

The floor there was set by the *problem* — an under-parameterised network ansatz, whose
approximation error put `‖F‖` at 2e-6, six orders of magnitude above the tolerance the solver was
asked for. That is not round-off, so no `eps(T)`-scaled tolerance can bound it, and the remedy is
one level up from the solver: raise `f_abstol` above the achievable residual, or improve the
approximation until its floor lies below the tolerance you need. Which is exactly what the report
now says.

- `SimpleSolvers.record_progress!` keeps, per solve, the residual as of the last iteration that
  counted as *progress* and the number of iterations since, so
  `SimpleSolvers.iterations_since_progress` measures how long the residual has been going
  nowhere. The reference is monotone — the best residual so far — so an iteration that undoes the
  previous one's progress does not reset the clock. Like `record_stall!` it is an increment
  rather than a predicate, so it owns its counter: an iteration that never records leaves it at
  zero, exactly as `stall_number` does. `SimpleSolvers.record_iteration!` is the one function
  that takes both measurements, from a single `residuals` call, and `solve!` calls it once per
  iteration.
- **The report is unconditional.** A solve that spends its whole budget without converging and
  without progressing over at least half of it — and over at least `F_STALL_REPORT_MINIMUM`
  iterations, below which the proportion is not evidence of anything — now says so, naming the
  residual it achieved, the
  tolerance it was asked for, how many iterations it went without improving, and that the iterate
  did *not* freeze — which is what distinguishes a floor of the problem from the round-off floor
  `max_stalls` reports. It replaces the bare iteration count rather than adding to it, and, like
  the stagnation message, is gated on `verbosity ≥ 1` and capped with `maxlog`. The bare count
  gained a `maxlog` too: it had none, so a time-stepping caller got one per step — and it is now
  also replaced by the *stagnation* message, which had been added alongside it since 0.10.0. The
  three are mutually exclusive, most specific first.
- **The stopping criterion is opt-in**, through two new `Options` fields: `f_stall_window`
  (default `0`, disabled) gives up after that many iterations without progress, and
  `f_stall_factor` (default `0.5`) is the drop that counts as progress. `SimpleSolvers.no_progress`
  is the predicate, gated on `!residual_small` exactly as `stalled_step` is, so giving up and
  converging remain mutually exclusive; `SimpleSolvers.isnotprogressing` reads it off the status.

The asymmetry is deliberate. The threshold is a policy, not a test: an iteration converging
linearly with rate ρ improves by ρ^W over a window W, so a window of 50 at the default factor
abandons every ρ > 0.986, and a `PicardSolver` on a stiff problem is slower than that. There is no
value that is right for every problem — so the *diagnosis*, which is consulted only about a solve
that has already spent its budget and therefore decides nothing, is free; while the *stopping*,
which can cut short a solve that would have succeeded, is the caller's to ask for. Set it once the
report has told you the floor is real.

Behaviour at default options is unchanged except for what is printed: no solve stops earlier than
it did, and `f_stall_window = 0` makes `no_progress` constant `false`.

### `NaN` was only half of it

Every guard the nonlinear solvers had against a trial iterate leaving the region where the
problem is representable tested `isnan`, and `isnan` is false for `Inf`. Issue #130 asks whether
that detection works, on the example `f(x) = 5|x-2|³ - 1/|x|` — whose singularity produces `Inf`
at `x = 0`, not `NaN`, so for its own fixture the answer was no. Nothing downstream of the guards
caught it either: `rfₐ > f_abstol_break` cannot, since `f_abstol_break` defaults to `Inf` and
`Inf > Inf` is false. The line searches had never had this gap — `check_anchor` has always tested
`isfinite` — and neither did the `Optimizer` removed in 0.8.0, which tested `isnan(f) || isinf(f)`.
The solvers were the only ones left looking for half the problem.

The distinction the fix turns on is that a non-finite **direction** and a non-finite **residual**
are not the same failure:

- A non-finite **direction is rejected**, as a `NaN` one always was. It cannot be recovered from:
  `nan_recovery!` damps by a factor and `Inf * nan_factor` is `Inf`, so the loop would spend its
  whole budget reproducing the same trial iterate. Rejecting it first is also what makes the
  damping sound — past that guard the direction is finite, so every halving really does shorten
  the step. Previously such a step was not rejected and not damped: the line search saw an
  infinite directional derivative, returned `LINESEARCH_NO_DESCENT`, and the solve made no
  progress without saying why.
- A non-finite **trial residual is damped**, exactly as a `NaN` one was. `nan_recovery!`,
  `report_dogleg_nan` and the dogleg trust-region rejection now all test `isfinite`.
- `havenan` is now `SimpleSolvers.havenonfinite` and tests all three residuals for finiteness.
  It is unexported and internal, so there is no deprecation.

Three defects next to it are fixed with it, the first of which is the serious one:

- **`residual_small` could report convergence on an overflowed solve.** It guarded only
  `isnan(r₀)`, so an infinite `‖F(x₀)‖` made the relative term `f_reltol * Inf = Inf` and the
  gate vacuously true — a residual of `1e10` counted as small, and the solve claimed success
  wherever it happened to land. An overflowed initial residual is no more a usable reference
  scale than an uninitialized one, and is now treated the same way.
- **`iterate_settled` counted an infinite step as a frozen iterate**, since `Inf ≤ ‖x‖·x_suctol`
  holds once `‖x‖` has overflowed too. An iterate that jumped to infinity has neither converged
  nor stagnated; it has broken down, which is what `havenonfinite` is for.
- **`nan_recovery!` at `nan_max_iterations = 0` left its cache untouched**, so the post-condition
  its docstring advertises — that `solution(cache)` and `value(cache)` hold this step's trial and
  its residual — was false. The `PicardSolver` step documents that it reads that cache instead of
  re-evaluating `F`, so it tested a stale residual and committed a stale iterate from the
  *previous* solver step. One trial is now always evaluated and the damping happens before the
  trial it is for, so the budget bounds the damping rather than the evaluation and the cached
  residual always belongs to the direction the cache holds. The Picard step additionally refuses
  to commit a trial whose *residual* is not finite, as `DogLegSolver` already did on radius
  underflow.

Behaviour on any solve that stays finite is unchanged, and the guards cost the same as before —
one pass over the residual either way. Where it changes, it changes in the direction of saying
so: a non-finite direction now raises `NonlinearSolverException` where it used to stall silently,
which a caller driving `solve!` in a time-stepping loop will see as an exception rather than as a
step that quietly achieved nothing. Downstream trajectories are bit-identical
(GeometricIntegrators × GeometricProblems, five integrators over 1000 steps, plus a collisional
initial condition with no root).

## [0.10.1]

Compile-time and allocation fixes. No behaviour change: the same messages, at the same `verbosity`
gates, with the same `maxlog` caps, and every test from 0.10.0 passes unedited.

### The defect

`SimpleSolvers.linesearch_warnings` is called from `solver_step!` on every iteration of every
solve, and its arguments are a `Linesearch` — which carries the closure types of its
`LinesearchProblem` — and a `NamedTuple` of parameters. It is therefore re-specialized for every
*problem* a solver is built for, and because the four `@warn` sites lived directly in its body,
so did all of their `Base.CoreLogging` and string-interpolation code: re-inferred and
re-codegen'd from scratch for each one. 0.9.2 had two short `@warn`s inline in `Backtracking`'s
`solve`; the 0.10.0 rewrite multiplied the message code roughly fivefold without changing where
it is specialized, so what had been a tolerable per-solver cost became the dominant one.

This is invisible in steady-state timings — a solve takes about a millisecond — and shows up
only where many solvers with distinct residual closures are built in one session. On
GeometricIntegrators' Runge-Kutta test suite, which builds one implicit integrator per tableau
(`Gauss(1)` … `Gauss(8)`, the `LobattoIII` and `Radau` families, `PGLRK`), it cost **76 s of the
suite's 145 s** — more than the whole rest of the line search and the Newton step together, and
96.9 % of the suite was compilation.

### Fixed

- The messages moved behind a `@noinline` function barrier,
  `SimpleSolvers.report_linesearch_status(status, name::Symbol, config::Options)`, which mentions
  no closure type and is therefore compiled exactly **once per element type for the whole
  session** instead of once per solver. `linesearch_warnings` remains the only place a line
  search emits messages, and is now a thin wrapper around the barrier plus the
  verbosity-2-gated `curvature_diagnostic` (which genuinely needs the `Linesearch` and the
  parameters, and costs nothing). The Runge-Kutta suite above went to **67 s**, with all 107
  assertions passing — recovering the entire regression. `nonlinear_solver_warnings` and
  `print_status` already had this shape, which is why they never cost anything.
- The same idiom was applied to every other message whose enclosing function is specialized on a
  merit closure or on a solver, so that no reporting site in the package grows with the number of
  solvers built: `curvature_diagnostic`'s message (`report_curvature_violation`, which is called
  from `linesearch_warnings` itself and was therefore the last per-solver message on that path),
  the two `bisection` messages (`report_bisection_nonconvergence`, `report_bisection_nobracket`),
  the three `DogLeg` ones (`report_dogleg_singular`, `report_dogleg_nan`,
  `report_dogleg_underflow`), `nan_recovery!`'s (`report_nan_direction`) and the `NewtonSolver`
  constructor's (`report_static_refactorize`). These messages are short, so the saving beyond
  `report_linesearch_status` is small; the point is that the whole package now has one idiom for
  reporting and no site that grows per solver.
- `linesearch_warnings` filters the two silent outcomes (`LINESEARCH_DECREASED`,
  `LINESEARCH_UNKNOWN`) before calling the barrier as well as inside it, so the path a healthy
  solve takes on every iteration does not make a call that would copy the 27-field `Options` for
  the callee.
- The `LINESEARCH_FLOOR` message's `αmin` clause and both wordings of the `LINESEARCH_EXHAUSTED`
  message are interpolated *inside* their `@warn` rather than into a temporary before the verbosity
  gate. Julia evaluates a `@warn` message only once the gate and `maxlog` have both passed, and a
  solve that cannot progress reports the same outcome on every iteration, so the temporaries were
  built and discarded once per iteration: 272 B per call for `EXHAUSTED` at *every* verbosity once
  its `maxlog = 3` was spent, and 704 B per call for `FLOOR` at `verbosity = 2`. Present since
  0.10.0.
- **A converged solve no longer touches the heap** for any solver or line search (up to
  ForwardDiff's own chunk-mode Jacobian, which allocates array wrappers above `n = 12`):
  - `BierlaireQuadratic`: 256 B per solve → 0. `_bierlaire_fit` both captured and mutated its
    evaluation counter in the merit closure, which boxes it and makes the `trials` of every
    `LinesearchStatus` built from the fit inferred-`Any`; and `triple_point_finder`'s
    `Union{Symbol,Tuple}` return had to be boxed. The bracketing loop moved into a type-stable
    `_triple_point_core` returning `(a, b, c, status)`, the same split `bisection` and
    `_bisection_core` already use; `triple_point_finder` keeps its documented return.
  - `StrongWolfe`: 256 B and 12 allocations per line search → 48 B and one. `wolfe_status` now
    takes the evaluation counter rather than capturing it, and the one-slot memo of φ and φ′ is a
    single holder instead of four `Ref`s. One allocation per line search remains by design: the
    closures that read the memo are handed to `_wolfe_zoom` and to the condition objects, so it
    cannot stay on the stack.

The barriers are internal and unexported. The contract is pinned down in both directions, and
without counting specializations: `test/linesearch_tests.jl` checks the *types* in
`report_linesearch_status`' signature, which bounds its specialization set to one per element type
by construction rather than sampling a couple of closures, and `test/logging_code.jl` scans the
lowered code of every reporter and every per-solver caller for `Base.CoreLogging`, asserting that
the messages are in the barriers and nowhere else. Since the verbosity gates moved with the
messages, the four reporters whose gate can be reached from a constructed problem — the two
`bisection` ones and `DogLeg`'s singular-Jacobian and NaN-merit ones — are additionally driven at
their documented verbosity and one below, so that a wrong gate in a future edit is not silent. The
message texts whose interpolation moved are pinned by text, and a silent reporter is asserted to
allocate nothing.

The allocation fixes are pinned by their *causes* rather than by byte counts, because the suite runs
under the `--check-bounds=yes` that `julia-actions/julia-runtest` passes, which inhibits the inlining
that keeps these closures off the heap and makes any fixed number meaningless: `test/lowered_code.jl`
scans lowered code for the `Core.Box` that a captured-and-mutated local produces, for every
line-search and solver kernel, and the returns whose boxing propagated are `@inferred`. Both hold
however the session was started. The end-to-end byte counts are asserted too, and skipped when the
session does not compile the package the way a caller does.

### Changed

- **The minimum Julia version is now 1.10** (was 1.6). 1.6 has been out of support for some time,
  CI has tested only 1.10 and newer for a while so the old bound was not exercised by anything,
  and both GeometricIntegratorsBase and GeometricIntegrators already require 1.10, so nothing
  downstream can be on less.

## [0.10.0]

Robustness release for the `Backtracking` line search. Downstream packages
(GeometricIntegrators, GeometricProblems, ChargedParticleDynamics) were flooded by

```
Backtracking line search did not satisfy the sufficient decrease condition within 1000
iterations. Returning the last trial step α = 2.220446049250313e-16.
```

— thousands of messages per test run — and could not silence it through the public API.

### The defect

`2.220446049250313e-16 == eps(1.0) == 1.0·0.5^52`, so the ladder always exited via its
`α ≤ eps` guard after ~53 trials; the "within 1000 iterations" in the message was never
true, and neither was the implied diagnosis. The sufficient decrease condition
`φ(α) ≤ φ(0) + c₁·α·φ'(0)` demands a decrease *proportional to* `α` with no round-off
allowance, so once `c₁·α·|φ'(0)|` falls below one ulp of `φ(0)` — at
`α* = eps/(4c₁) ≈ 5.55e-13` for the `‖F‖²` merit of a Newton step — the right-hand side
rounds back up to `φ(0)` and the accept/reject decision is made by *rounding alone*. Two
failure modes followed:

- when the trial point stopped moving above `α*`, the condition "succeeded" at `α ≈ 2.3e-13`
  purely because the frozen merit tied a rounded right-hand side. The returned step did not
  move `x` at all, and **nothing was reported**;
- otherwise the ladder ran to `α = eps` and emitted the warning above.

Either way the outer iterate froze, the solve spun to `max_iterations`, and a second
misleading warning (`"Solver took 1000 iterations."`) followed. The underlying user-visible
cause is an `f_abstol` below the residual's own round-off floor — unsatisfiable, and *not*
rescued by `f_reltol`, which is anchored at `‖F(x₀)‖` and therefore becomes *tighter* the
better the initial guess is.

### Added

- `LinesearchStatus`, `LinesearchOutcome` (`LINESEARCH_DECREASED`, `LINESEARCH_FLOOR`,
  `LINESEARCH_EXHAUSTED`, `LINESEARCH_NO_DESCENT`, `LINESEARCH_STATIONARY`,
  `LINESEARCH_UNKNOWN`) and `solve_with_status` are exported; the accompanying predicates and
  accessors `issufficient`, `isfloor`, `steplength`, `outcome` and `trials` are *not*, since
  they are generic names a package doing `using SimpleSolvers` may want for itself — reach them
  as `SimpleSolvers.steplength` and so on. `solve_with_status`
  returns the step length *plus* why the search stopped, which the step length alone cannot
  express — a tiny `α` may be the right answer or all that is left after the merit turned out
  to be irreducible. All six built-in methods report a genuine outcome (see the contract section
  below); the generic `LINESEARCH_UNKNOWN` fallback remains for user-defined methods.
  `SimpleSolvers.linesearch_warnings` is now the single place a line search emits messages.
- `Backtracking` gained a `τ_ulps` keyword (default 4): the round-off *resolution* of the merit,
  `τ = τ_ulps·ulp(φ(0))`, from which the smallest *informative* step
  `αmin = τ/(c₁|φ'(0)|) = 2·τ_ulps·α*` is derived. Because `αmin` is a factor `2·τ_ulps` above
  `α*`, the search does not enter the region where the test is decided by rounding — unless
  `backtracking_αmin`'s upper clamp at `√eps(T)` binds, which it does for a very flat merit in
  double precision and for essentially any merit in `Float16`. Trial steps below `α*` are safe
  there because the condition reduces to plain monotonicity (see below), and such an accept is
  classified `LINESEARCH_FLOOR`. `τ_ulps = 0` recovers the exact condition. New (unexported)
  `armijo_tolerance`, `backtracking_αmin`, `backtracking_interpolation`, and the constants
  `DEFAULT_ARMIJO_τ_ULPS` and `BACKTRACKING_SHRINK_MIN`.
- `SufficientDecreaseCondition` gained a keyword-only `τ` field (default `zero(T)`, i.e. the
  exact former condition) and a two-argument call form `sdc(α, fα)` taking a merit value the
  caller already computed, so neither `Backtracking` nor `StrongWolfe` evaluates the merit twice
  per trial. `StrongWolfe` keeps `τ = 0`. The condition is
  `fα ≤ min(f₀, f₀ + cαd₀ + τ)`: the allowance may reduce the decrease *demanded*, but it never
  accepts a step whose merit exceeds `f₀`.
- `armijo_ulps(T[, c₁])` makes the round-off resolution **precision-aware**, and is what every
  method now uses in place of the bare `DEFAULT_ARMIJO_τ_ULPS`. `τ` must be at least ~an ulp of
  `φ(0)` to recognise a merit at its round-off floor, and far below the `2c₁·φ(0)` the condition
  demands at `α = 1` to leave that condition meaningful; the two are compatible only while
  `eps(T) ≪ 2c₁`. That holds by ~10 orders of magnitude in `Float64` and by a factor 400 in
  `Float32`, so the nominal 4 ulps stands in both and their behaviour is bit-identical to before.
  It fails outright in `Float16`, where `eps(T) = 9.8e-4` already exceeds `2c₁ = 2e-4`: the
  nominal `τ` was *twenty times* the demanded decrease, which degenerated the Armijo test to
  plain monotonicity at every `α` and — worse — classified a genuine two-ulp decrease as
  `LINESEARCH_FLOOR`, which `solver_step!` feeds to `flag_stall!`, so a *converging* half-precision
  solve could be reported as stagnated. The cap (`ARMIJO_τ_DEMAND_FRACTION = 0.01`, i.e. `τ` may
  distort the demanded decrease by at most one percent) drops `Float16` to ~2e-3 ulps, in effect
  the exact condition. Nothing is lost: the floor is still detected without `τ`, both by the
  condition degenerating to `φ(α) ≤ φ(0)` at a small enough trial step and by `Backtracking`'s
  two-consecutive-bit-identical-merits check. `Backtracking` applies the cap in its inner
  constructor, so every path in — including `change_precision` and an explicitly oversized
  `τ_ulps` — gets a resolution the element type can support.
- `Options` gained `linesearch_max_iterations` (default `linesearch_iterations(T)`: 60 for
  `Float64`, 31 for `Float32`, 18 for `Float16`) and `max_stalls` (default `MAX_STALLS = 2`).
- `SimpleSolvers.with_config(ls, config)` returns a `Linesearch` with the problem and method of
  `ls` but different `Options`.
- `NonlinearSolverStatus` is exported; `SimpleSolvers.isconverged`, `isstalled` and
  `status(solver, state)` are not, `status` least of all — a downstream package that does
  `using SimpleSolvers` and defines its own `status` would get a method-definition error rather
  than a shadowing warning. `solve!` returns `x`, not a status, so
  `SimpleSolvers.status(solver, state)` is how a caller inspects the outcome — in particular
  whether a solve *converged* or merely *stagnated* at the residual floor.
- `SimpleSolvers.residual_small`, `iterate_settled`, `stalled_step`, `record_stall!`,
  `flag_stall!`, `stall_number` and `needs_refresh`.

### Changed (behavioural)

- **`Backtracking` no longer decides the sufficient decrease condition by rounding.** It now
  checks the anchor up front (a non-descent or non-finite `φ'(0)` returns the caller's `α`
  immediately instead of shrinking 53 times to find out, consistent with `StrongWolfe`; a
  stationary `φ'(0) = 0` is benign and reported only at `verbosity ≥ 2`), stops at `αmin`
  rather than `eps`, detects a bit-frozen merit, and shrinks by safeguarded quadratic/cubic
  interpolation confined to `[0.1α, p·α]` instead of a fixed factor `p`. `Backtracking.p` is
  therefore now an *upper bound* on the shrink factor, so the trial sequence is pointwise never
  longer than before. Warnings report the true trial count and the true reason. The
  round-off-floor and stationary-anchor outcomes are reported only at `verbosity ≥ 2` — both are
  the *expected* final state of a converged solve, so warning about them at the default
  verbosity would mean warning about success (measured on GeometricProblems, gating them at
  `verbosity ≥ 1` newly surfaced one message each on three previously-silent solves). The
  remaining outcomes are rate-limited with `maxlog`.
- **A line search now shares its solver's `Options`.** Every solver used to build its
  `Linesearch` with an `Options` constructed from nothing but defaults, so
  `NewtonSolver(…; verbosity = 0)` did *not* silence the line search (downstream packages had
  to swallow the messages with a `NullLogger`) and a user-supplied iteration budget never
  reached the inner ladder. `config(linesearch(s)) === config(s)` now holds for `NewtonSolver`,
  `PicardSolver` and `DogLegSolver`, in every constructor arity.
- **`max_iterations` no longer doubles as the line-search budget.** It bounds only the outer
  nonlinear iteration; the `Backtracking` ladder, the `StrongWolfe` bracketing and zoom phases
  and `bisection` are bounded by `linesearch_max_iterations`. The default therefore drops from
  1000 to 60 for `Float64` — unobservable, since all of these loops terminate on their own
  floor long before either bound. `Quadratic` and `BierlaireQuadratic` are bounded by the same
  field: the (unexported) `max_number_of_quadratic_linesearch_iterations` and the constants
  `MAX_NUMBER_OF_ITERATIONS_FOR_QUADRATIC_LINESEARCH[_SINGLE_PRECISION]` are removed, so their
  budget is now settable by the user like every other line search's.

  This is the one place where the split *is* observable, and it is an improvement. Their budget
  rises from 20 to 60 (`Float64`) and from 5 to 31/18 (`Float32`/`Float16`), which lets them
  resolve the one-dimensional subproblem further. Measured over 300 random starting points on
  the `exp(x)(x³ − 5x² + 2x) + 2` root-finding fixture with `BierlaireQuadratic` as the
  `NewtonSolver` line search, the `Float64` 95th-percentile distance from the root drops from
  1.9e6 to 0.5 `eps` with a fresh Jacobian and from 6.0e6 to 1.6 `eps` with `refactorize = 5`
  (median 8599 → 0.50). Note that most of that gain comes from fixing the single-point stall
  described below, not from the budget itself: the intermediate figures (9.7e3 and 8.5e4 `eps`)
  were solves hitting the raised cap through the stall.
- **A stagnating solve now stops after `max_stalls` steps instead of running to
  `max_iterations`.** A step is *stalled* when it did not move the iterate while the residual is
  not small — the exact logical complement of the existing `x_converged` gate, so stagnation
  and convergence are mutually exclusive by construction. This is diagnosed from the step
  actually taken, so it also covers a `Static` step along an underflowed direction, a collapsed
  `DogLeg` trust-region radius and a locally expanding `Picard` map; a line search that reports
  `LINESEARCH_FLOOR` or `LINESEARCH_NO_DESCENT` additionally flags it one iteration earlier.
  A counter (rather than a one-shot stop) is used because the step *after* a stalled one is
  deliberately attempted under better conditions — see the next entry. `max_stalls = typemax(Int)`
  restores the old behaviour. The two misleading warnings are replaced by one message naming the
  achieved residual, the requested tolerance and both remedies, rate-limited with `maxlog` so
  that a caller looping `solve!` per time step does not see it once per step.
- **A stalled step forces a fresh Jacobian on the next step.** `maybe_refactorize!` gained a
  `stalled` keyword, supplied by `solver_step!` from the new `SimpleSolvers.needs_refresh(state)`.
  Without it, a quasi-Newton solver with `refactorize = r` would rebuild the same direction from
  the same stale Jacobian for up to `r - 1` further steps and reproduce the same negligible step,
  so `max_stalls = 2` could give up at the second of those for a reason a fresh Jacobian would
  have fixed. Now the second consecutive stall is one a freshly evaluated Jacobian did *not* fix,
  which makes `max_stalls = 2` conclusive for every `refactorize` rather than only for
  `refactorize = 1`. `DogLegSolver` folds the same signal into its existing `force_refresh`.
- **`LINESEARCH_NO_DESCENT` is acted on rather than only reported.** An ascent anchor is a
  stale-Jacobian symptom whose remedy is to refresh the Jacobian, so `solver_step!` now leaves the
  iterate where it is — moving along a direction the line search has rejected outright would only
  make the retry start from a worse point — and records a stall, which triggers exactly that
  refresh. Previously the full step was taken along the ascent direction, the iterate moved (so
  nothing counted it as a stall), and the solve ran to `max_iterations`. The line search itself
  still returns `α > 0` as its contract requires; whether to use it is the caller's decision.
- **`solve!` tests the stopping criteria before the first step.** An initial guess that already
  satisfies them is no longer perturbed by a full solver step (Jacobian, factorization, and a
  line search asked to improve an already-exact residual). Only the absolute branch is
  reachable at iteration 0, so this fires exactly when `‖F(x₀)‖ ≤ f_abstol`; with the default
  `f_abstol = 0` that means an exact root. Two consequences: with `min_iterations = 0` and
  `f_abstol > 0` an already-satisfying guess now takes **zero** steps instead of one, and
  `f_abstol_break` can now fire before any step. `min_iterations ≥ 1` is unaffected.
- The line search no longer re-evaluates the merit at the `α = 0` anchor: `solver_step!` passes
  the residual the solver has already computed as `params.φ₀`, saving one `F` evaluation per
  solver step. A caller driving `solver_step!` by hand from a state whose `value` is stale must
  not supply it.
- `StrongWolfe` no longer evaluates the merit twice per trial step. Both its expansion loop and
  its zoom phase called the one-argument `sdc(α)` immediately after computing `φ(α)`, and the
  status construction re-evaluated `φ` at the accepted step; all three now reuse the value in
  scope (`_wolfe_zoom` returns `(α, φ, n)`). For a `NonlinearSolver` each of those was a full
  residual evaluation.
- `LinesearchStatus.trials` is now the true merit-evaluation count for every method.
  `Bisection`, `Quadratic` and `BierlaireQuadratic` reported a hardcoded `0`, so the
  round-off-floor message read "in 0 trial step(s)", and `StrongWolfe` counted only its expansion
  loop. `LinesearchStatus.αmin` remains a shrinking-ladder quantity and is documented as
  `zero` — "not applicable" — for the bracketing and minimising searches, which no longer print
  it.
- `Base.show(::Backtracking)` includes `τ_ulps`; `Base.show(::NonlinearSolverStatus)` appends a
  `stalls` line only when it is nonzero, so the printout of a fresh status is unchanged.
- Passing `linesearch = …` to `PicardSolver`/`DogLegSolver` is still an error, but now raises a
  `MethodError` rather than an `Options` error.

### The line-search contract (standardised across all six methods)

Investigating a crash in `BierlaireQuadratic` (below) showed the six line searches differing
along eight independent axes — only some of them principled. Every method reached through
`solve`/`solve_with_status` now guarantees:

1. **It never throws.** A situation it cannot handle is *reported*, never raised: a line search
   must not abort the enclosing solve. `triple_point_finder`, `bracket_minimum` and
   `bracket_minimum_with_fixed_point` return `nothing` instead of calling `error(...)` (four
   sites), and each method maps that onto a `LinesearchOutcome`.
2. **It returns `α > 0`.** Never the `α = 0` anchor (which freezes the outer iterate) and never a
   negative step. `Quadratic` and `Bisection` used to return negative steps — measured at up to
   `α = -3` for `Bisection`, in 49 of ~3750 calls with `refactorize = 5` — because both inherit a
   direction flip from `bracket_minimum`. That was emergent rather than designed: `α` scales a
   direction that has already been chosen, an ascending anchor arises *only* from a stale Jacobian
   (zero occurrences with `refactorize = 1` over 3727 calls), and the correct response to
   staleness is to refresh the Jacobian, which `maybe_refactorize!` already does. The flip stays
   in `bracket_minimum` itself, which is a general-purpose minimum bracketer; only the line-search
   layer restricts it. Verified: 0 negative steps in ~45 000 calls, down from 49.
3. **All six implement `solve_with_status`; `solve` is a thin wrapper.** Previously only
   `Backtracking` did, which meant `solver_step!`'s `isfloor(lsstatus) && flag_stall!(state)` was
   dead code for five of six methods — stall detection now works whatever line search is chosen.
   This also collapses six inline warning sites into `linesearch_warnings`, giving one message
   site and one verbosity policy (`Bisection` warned at `≥ 1` where `BierlaireQuadratic` warned at
   `≥ 2` for the same class of event).
4. **A non-finite or ascending anchor is reported, not assumed away** — the new (unexported)
   `check_anchor` is the single definition of that policy, used by every method. This closes an
   `AssertionError` in `StrongWolfe`: a `NaN` derivative is not `≥ zero(T)`, so it slipped past
   the descent check into `SufficientDecreaseCondition`'s `@assert !isnan(d₀)` and aborted the
   solve. `StrongWolfe` could also return `α = 0` from its zoom phase, despite a comment claiming
   otherwise.
5. **Cost is bounded independently of the merit's scale.**

Deliberately *not* standardised: the meaning of the input `α` and what each method guarantees
about the step, because there are two families — condition-satisfying and `α`-relative
(`Backtracking`, `StrongWolfe`, `Static`) versus minimising and `α`-independent (`Bisection`,
`Quadratic`, `BierlaireQuadratic`, which check no Wolfe condition). Documented on
`LinesearchMethod`.

`LINESEARCH_DECREASED` is documented as "the merit decreased by more than `τ`" rather than
"satisfies the `SufficientDecreaseCondition`", since the minimising searches never test one.

### Fixed: `BierlaireQuadratic` could abort a solve, or stall until its budget ran out

**The crash.** `triple_point_finder` raised
`The function f must be decreasing at 0.0` — 3 of 300 random starts — aborting the whole solve.
Two distinct causes, both measured: an **ascent anchor** (`φ′(0) = +2.06e-15` for `Float64`,
`+1.10e-02` for `Float32`, both with `refactorize = 5`), because the entry point never checked
`φ′(0)` while claiming in a comment that the `α = 0` anchor was "guaranteed decreasing"; and a
**merit flat to round-off** (`φ(δ) − φ(0) = 0` exactly, `Float32`, `refactorize = 1`). The
response was wrong for both: halving `δ` five times and then erroring. Halving answers an
*overshoot*; for an ascent direction nothing helps, and for a round-off-limited probe a smaller
`δ` is strictly less informative. `triple_point_finder` now halves only when the rise exceeds
`4 eps(f(x₀))`, so a flat merit costs 2 evaluations instead of 12 followed by a throw.

**The stall.** `φ(α) = c·(α−1)²` cost 15 / **70** / 14 merit evaluations at `c` = 1 / 1e-6 /
1e-12 — a 5× swing from a pure rescaling, with the middle case exhausting its budget. It was
stuck on a single point, evaluating `α = 0.99999999999999911` thirty-plus times in a row: when
`χ` lands on `b`, both branch tests are false, so the triple becomes `c ← b` with `b` unchanged,
the next fit is degenerate, and the `(a+c)/2` fallback plus the anti-stalling shift map back onto
`b` while `c − a` never shrinks. `shift_χ_to_avoid_stalling` exists to prevent exactly this and
does not. The fit now bisects the wider sub-interval whenever `χ` coincides with a bracket point,
making `c − a` contract strictly every iteration. Cost is now 15/16/16/15/15 across merit scales
`1e-12 … 1e12`, and no `α` is evaluated more than twice.

This was also the cause of the poor accuracy that the previous entry attributed to the raised
iteration budget: those solves were hitting the cap, not converging poorly. Over 300 random
starts the `Float64` 95th-percentile distance to the root improves from **9.7e3 to 0.5 `eps`**
with a fresh Jacobian and from **8.5e4 to 1.6 `eps`** with `refactorize = 5` (median 8599 → 0.50),
so the `tolfac` on the two `BierlaireQuadratic` fixture rows is back to the 2 `eps` every other
method meets.

The merit-space comparisons in the `BierlaireQuadratic` termination test now use the shared
round-off allowance `τ = armijo_tolerance(φ₀, τ_ulps)` instead of `ε`. One absolute constant used
to govern three incommensurable quantities — a bracket width in `α` and two merit differences.
This is correctness-neutral (the returned `α` was already scale-invariant, and the `α`-space width
term is the binding one); it removes a latent hazard rather than fixing observed behaviour.

Internal `solve` overloads that were algorithm steps rather than entry points
(`solve(ls, a, b, c, params, n)`, `solve(ls, α₀, params, n)`, `solve(ls, α₀, α₁, params)`) are now
private helpers, so `solve`/`solve_with_status` is unambiguously the public entry point.

### Changed (breaking)

- `Quadratic` and `Bisection` no longer return negative steps. The test that pinned the old
  behaviour asserted `solve(ls, 0.0) ≈ -1.0`. A result that is still non-positive after the
  retry from the `α = 0` anchor is reported as `LINESEARCH_FLOOR` (not `LINESEARCH_NO_DESCENT`,
  which would contradict the anchor check that already passed).
- `bracket_minimum` and `bracket_minimum_with_fixed_point` return `Union{Nothing,…}` instead of
  throwing, and `triple_point_finder` returns `Union{Symbol,…}`. **`bracket_minimum` is
  exported**, so `a, b = bracket_minimum(f, x)` on an unbracketable `f` now raises
  `MethodError: iterate(::Nothing)` where it used to raise a descriptive error, and code that
  relied on the throw to detect failure now proceeds silently. The other two are not exported.
  `triple_point_finder` distinguishes its two failures — `:flat` (the merit does not resolve a
  decrease, mapped to `LINESEARCH_FLOOR`) from `:unbracketable` (there *is* a decrease that
  could not be bracketed, mapped to `LINESEARCH_EXHAUSTED`) — because conflating them makes the
  outer iteration count a *descending* merit as stagnation.
- Removed the unexported `max_number_of_quadratic_linesearch_iterations` and the constants
  `MAX_NUMBER_OF_ITERATIONS_FOR_QUADRATIC_LINESEARCH` and its `_SINGLE_PRECISION` sibling; both
  quadratic searches now use `Options.linesearch_max_iterations`.
- Removed the `solve(::Linesearch{<:Bisection}, α₀, α₁, params)` overload and the
  `BierlaireQuadratic` `solve(ls, α₀, params, iteration_number)` overloads, which made the
  public `solve` name ambiguous; `solve`/`solve_with_status` with a single `α` is the entry
  point. The private `_bisect_on`, `_bierlaire_fit` and `_quadratic_search` replace them.
- `DEFAULT_ARMIJO_τ_ULPS` and `armijo_tolerance` moved from `linesearch/backtracking.jl` to
  `base/options.jl` so that `bracketing/` can use them, and `armijo_tolerance`'s second argument
  widened from `::T` to `::Real`. Both are unexported.
- `assess_convergence` returns a **4-tuple** `(x_converged, f_converged, f_increased, stalled)`.
  Existing `a, b, _ = assess_convergence(…)` destructurings keep working unchanged.
- `NonlinearSolverStatus` gained the fields `stalls::Int` and `stalled::Bool`;
  `NonlinearSolverState` gained `stalls::Int` and `stallflag::Bool`. Positional construction of
  either changes accordingly.
- `Backtracking` gained a fifth field `τ_ulps::T`. The positional inner constructor defaults
  it, so `Backtracking{T}(α₀, c₁, c₂, p)` still works.

## [0.9.2]

### Fixed

- `Quadratic`: a `Float16` fix in the quadratic line search.

## [0.9.1]

### Changed (breaking)

- The default line search of `NewtonSolver(x, F, y; …)` is `Backtracking` again, reverting the
  change to `StrongWolfe` made in 0.9.0. Pass `linesearch=StrongWolfe(T)` to opt in to the
  strong Wolfe conditions.

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
- The `QuasiNewtonSolver` type alias and export (it was just
  `NonlinearSolver{T,Newton{false}}`, i.e. the same solver as `NewtonSolver` with a
  different `refactorize` default). There is now a single `NewtonSolver` type
  (`NonlinearSolver{T,Newton}`); build a quasi-Newton solver with
  `NewtonSolver(…; refactorize=n)` (or via `NonlinearSolver(QuasiNewton(n), …)`).
  ([issue #149](https://github.com/JuliaGNI/SimpleSolvers.jl/issues/149))
- The error-swallowing fallbacks `initialize!(x...)`, the 1-arg
  `solver_step!(::NonlinearSolver)`, and the generic two-arg `Gradient` functor;
  unsupported calls now raise a proper `MethodError`.
- The `Options.g_restol` field (and its `g_restol(::Options)` accessor). Its role — the
  residual tolerance in the convergence check — is now filled by `f_reltol`
  (defaulting to `√eps(T)`, `g_restol`'s former value), which additionally scales with
  the initial residual. `Options(; g_restol=…)` is no longer a valid keyword.

### Changed (breaking)

- The concrete nonlinear-solver method types were renamed to drop the `Method` suffix:
  `NewtonMethod` → `Newton`, `QuasiNewtonMethod` → `QuasiNewton` and
  `PicardMethod` → `Picard`. The old names are gone; update
  `NonlinearSolver(NewtonMethod(), …)` call sites to `Newton()` (etc.).
- `Newton` no longer carries a type parameter (was `Newton{QT}` with
  `QuasiNewton = Newton{false}`): the Newton/quasi-Newton distinction is entirely
  captured by the existing `refactorize::Int` field, so `Newton` is now a plain
  struct with `Newton(refactorize=1)` (mirroring `DogLeg`). `QuasiNewton` is kept
  as a convenience constructor `QuasiNewton(refactorize=5) = Newton(refactorize)`.
  `Newton{true}`/`Newton{false}` no longer parse.
  ([issue #149](https://github.com/JuliaGNI/SimpleSolvers.jl/issues/149))
- The default line search of `NewtonSolver(x, F, y; …)` changed from `Backtracking`
  to `StrongWolfe` (the only line search that genuinely enforces the strong curvature
  condition). Pass `linesearch=Backtracking(T)` to restore the previous behavior.
  (`DogLegSolver` is a trust-region method and does not run the stored line search, so
  its default is unaffected; `PicardSolver` takes no line search.)
  **Reverted in 0.9.1** — the default is `Backtracking` again; see below.
- `CurvatureCondition`'s `mode` is now a positional `Val{:Standard}()`/`Val{:Strong}()`
  argument (was a runtime `mode::Symbol` keyword) — inference-stable.
- `DogLegSolver` now uses a carried, ρ-based trust-region radius (N&W Alg. 4.1; new
  `DogLegCache` `trust_radius[!]` accessors) and `solver_step!(::DogLegSolver)` no
  longer takes a `Δ` keyword. (Like `Newton`, DogLeg still accepts a `refactorize`
  keyword, `refactorize > 1` reusing the Jacobian and its factorization between steps
  for a quasi-Newton-style dogleg method.)
- `DogLegSolver` no longer accepts a `linesearch` keyword. DogLeg is a trust-region
  method: its `solver_step!` sets the step length via the trust-region radius and never
  consults a line search, so the keyword was silently ignored (and the inherited
  `Static`-line-search warning falsely implied it mattered). Passing `linesearch=…` is
  now an error rather than a no-op; the structurally mandatory field is filled with a
  trivial `Static` step. Same fix as the `QuasiNewton`/`Picard` silently-ignored-keyword
  cases.
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
  `print_jacobian` (and their `NewtonSolver` forwarders) gained an
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

---

## Open Issues

Problems that surfaced while working on a release but were **not** fixed in it. Each entry says
what is wrong, how it was established, and why it was left. When one is fixed, move it out of this
section and into the release that fixed it.

### Reported by GeometricOptimizers.jl, not addressed in 0.12.0

- **`Bisection` can converge onto a *maximum*.** The 0.12.0 fix closed the route by which a
  *failed* bracket became `LINESEARCH_FLOOR`; it did not touch the route by which a *successful*
  one does. `bracket_minimum` brackets a minimum in **value**, which on a non-convex ray can
  enclose several stationary points, and the bisection of ``\varphi'`` inside it may settle on any
  of them. Landing on a maximum is not caught: the step is classified by the merit like any other,
  so it is `LINESEARCH_DECREASED` when it happens to improve ``\varphi`` and `LINESEARCH_FLOOR`
  when it does not — the same overclaim, reached by a different path. GeometricOptimizers observed
  exactly this (`_BFGS` + `Bisection` + `Geodesic`: `LINESEARCH_FLOOR` with ``\varphi(1) =
  \varphi(0)``, and the step taken regardless). A second-order check at the located root, or
  rejecting a root with ``\varphi'' < 0`` and re-bracketing, would close it. Left out because it
  changes what the method *searches for*, not merely what it *reports*, which is a larger change
  than the reporting defect that prompted 0.12.0.
- **`store_trace`, `show_trace` and `extended_trace` have no readers.** `grep` over `src/` finds
  them only in `Options` itself — they are accepted, printed by `show(::Options)`, and then
  ignored, so a caller who sets one gets silence rather than a trace or an error.
  GeometricOptimizers implements `store_trace` itself for this reason. Either implement them or
  remove them; both are breaking, so neither belongs in a release driven by something else.

### Introduced or left standing by 0.12.0

- **The status can misreport ``\varphi(\alpha)`` when the merit cannot be bracketed at all.** When
  `bracket_minimum` returns `nothing`, `Bisection`'s `solve_with_status` returns a
  `LinesearchStatus` whose `φ` field is filled with ``\varphi(0)`` rather than the merit at the
  ``\alpha`` it hands back. Reproducer: ``\varphi(\alpha) = 1 - \alpha`` descends forever, and the
  status reports `α = 1.0, φ(α) = 1.0` where the true value is `0.0`. Nothing acts on that field —
  the outcome is decided before it is filled — but `linesearch_exhausted_reason` interpolates the
  difference into its message, where it reads as "the merit did not change" when in fact it was
  never measured. Pre-existing, not a regression. The honest fix is to evaluate the merit at the
  returned step, which costs one evaluation on a path that is already a failure.
- **A third-party `LinesearchMethod` still reports from inside a solve.** The layering that 0.12.0
  established — a program calls `solve_with_status` and gets no messages — holds because all six
  built-in methods implement `solve_with_status` directly. The generic fallback does not: it calls
  `solve`, which calls `linesearch_warnings`. So a downstream method that implements only `solve`
  reinstates the per-iteration message the release removed, and now without a `maxlog` to bound it.
  Inverting the fallback (have the default `solve` call `solve_with_status`) would fix it, but the
  default `solve` is also the documented extension point, so this needs a decision about which of
  the two a method is expected to define.
- **`should_report!` does not promise every occurrence, and its counters are global.** A repeating
  diagnosis is reported on occurrences 1, 2, 4, 8, 16 … — occurrence 10 is silent. This is the
  intended trade (see its docstring) and not a defect, but it *is* a behaviour a caller can be
  surprised by, and there is no `Options` knob to opt out of it: the only controls are
  `verbosity = 0` and `reset_warning_counts!`. The counters are also process-global and shared
  across unrelated solvers, which is deliberate — a per-solver counter would reset on every step of
  a loop that rebuilds its solver, which is exactly the flood the cap exists for — but it means two
  independent solves can suppress each other's reports.
