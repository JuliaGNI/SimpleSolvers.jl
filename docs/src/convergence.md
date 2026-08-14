# Convergence and Stopping

Two separate verdicts are reached in a solve, and this page documents both, together with the one
channel that connects them:

1. a **line search** decides, for a single direction, whether it found a step that decreases the
   merit — and if not, *why not*. Its verdict is a [`LinesearchOutcome`](@ref) carried in a
   [`LinesearchStatus`](@ref);
2. the **outer iteration** decides whether the solve has converged, stagnated, or has to be given
   up on. Its verdict is a [`NonlinearSolverStatus`](@ref).

They are deliberately not the same question. A line search that reports failure does not mean the
solve failed, and a solve that converged will normally have a line search reporting the merit's
round-off floor on its last step. Section [How a line search's verdict reaches the
solver](@ref "How a line search's verdict reaches the solver") is where the two meet.

!!! info "Stopping is not converging"
    [`SimpleSolvers.meets_stopping_criteria`](@ref) answers *"should the loop end?"* and
    [`SimpleSolvers.isconverged`](@ref) answers *"is the answer good?"*. Most of the ways a solve
    ends are not convergence. `solve!` returns the solution either way, so a caller that needs to
    know which happened uses [`solve_with_status!`](@ref).

## Part 1 — the line search

### The round-off resolution ``\tau``

Every criterion below is expressed against ``\tau``, the resolution of the merit:

```math
\tau = n \cdot \mathrm{ulp}(\varphi(0)), \qquad n = \texttt{armijo\_ulps}(T),
```

see [`SimpleSolvers.armijo_tolerance`](@ref) and [`SimpleSolvers.armijo_ulps`](@ref). It exists
because a decrease smaller than the last bits of ``\varphi(0)`` is not a decrease that was
*measured* — it is one that rounding produced. ``\tau`` is what separates
`LINESEARCH_DECREASED` from `LINESEARCH_FLOOR`, and it is precision-aware: the nominal
[`SimpleSolvers.DEFAULT_ARMIJO_τ_ULPS`](@ref) of 4 is capped so that ``\tau`` can never amount to
more than [`SimpleSolvers.ARMIJO_τ_DEMAND_FRACTION`](@ref) of the decrease the sufficient decrease
condition demands, which matters in `Float16`.

### The six outcomes

A [`LinesearchOutcome`](@ref) is what [`solve_with_status`](@ref) reports. In terms of the returned
step ``\alpha`` and the merit ``\varphi`` there:

| outcome | the test that produced it | benign? |
|---|---|---|
| `LINESEARCH_DECREASED` | ``\varphi(\alpha) \leq \varphi(0) - \tau`` | yes |
| `LINESEARCH_FLOOR` | no trial step changed the merit by more than ``\tau`` | no |
| `LINESEARCH_EXHAUSTED` | the merit *does* vary by more than ``\tau``, but no acceptable step was found | no |
| `LINESEARCH_NO_DESCENT` | ``\varphi'(0) > 0``, or ``\varphi(0)``/``\varphi'(0)`` not finite | no |
| `LINESEARCH_STATIONARY` | ``\varphi'(0) = 0`` | yes |
| `LINESEARCH_UNKNOWN` | the method evaluated no merit and so established nothing | yes |

"Benign" is [`SimpleSolvers.isbenign`](@ref), and it is what
[`SimpleSolvers.linesearch_failures`](@ref) counts the complement of. Note that
`LINESEARCH_FLOOR` counts as a *failure* there even though it is the expected final state of a
converged solve — whether it matters is the outer iteration's call, and that is the layer that
makes it.

Two distinctions in that table carry real weight and are worth stating explicitly, because
conflating either one makes a solve report the wrong cause:

- **`FLOOR` is not `EXHAUSTED`.** `LINESEARCH_FLOOR` asserts that *no* line search can make
  progress along this direction — the merit is irreducible here. `LINESEARCH_EXHAUSTED` asserts
  only that *this* search did not find a step, leaving open that one exists. The outer iteration
  acts on the first (it counts towards `max_stalls`) and not on the second, so a search that
  reports a floor it has not established makes a healthy solve look stagnant.
- **`NO_DESCENT` is not a tolerance problem.** It says the direction itself is wrong, which points
  at the [`Jacobian`](@ref): a stale one under `refactorize > 1`, a nonzero
  `regularization_factor`, or an inexact linear solve. The remedy is to refresh the Jacobian, and
  [`SimpleSolvers.solver_step!`](@ref) does exactly that rather than shrinking a step that cannot
  help.

!!! info "What a `DECREASED` does and does not claim"
    It claims a decrease exceeding ``\tau``, and nothing more. [`Backtracking`](@ref) and
    [`StrongWolfe`](@ref) additionally verify their Wolfe condition before returning; the minimising
    searches ([`Bisection`](@ref), [`Quadratic`](@ref), [`BierlaireQuadratic`](@ref)) approximate the
    line minimiser and test no such condition. The ``\tau``-exceeding decrease is the guarantee they
    all share.

### Which stationary point a minimising search finds

[`Bisection`](@ref) drives on the *sign* of ``\varphi'``, and a bisection converges to whichever
crossing the sign at its left endpoint selects — a sign that is invariant under the halving. From
``\varphi'(\mathrm{lo}) < 0`` the interval shrinks onto a ``-\to+`` crossing, a **minimum**; from
``\varphi'(\mathrm{lo}) > 0`` onto a ``+\to-`` crossing, a **maximum**.

Nothing in [`bracket_minimum`](@ref) rules the second out: it brackets a minimum in *value*,
sampling ``\varphi`` and never ``\varphi'``, so on a non-convex ray its interval can enclose several
stationary points and its left endpoint can sit past one of them. The orientation is therefore
checked, and repaired by bisecting ``[0, \mathrm{lo}]`` instead — an interval that brackets a
minimum by construction, since [`SimpleSolvers.check_anchor`](@ref) has established
``\varphi'(0) < 0``. See `SimpleSolvers._bisect_for_minimum`.

This matters for the criteria on this page rather than only for the answer. A step at a maximum is
classified by the merit like any other, so an unrepaired search reported `LINESEARCH_FLOOR` whenever
that step failed to improve ``\varphi`` — and `LINESEARCH_FLOOR` is a claim about the *direction*,
which the outer iteration acts on.

### The anchor policy

Before any method searches, it validates the ``\alpha = 0`` anchor through the single shared
[`SimpleSolvers.check_anchor`](@ref):

- ``\varphi(0)`` or ``\varphi'(0)`` not finite, or ``\varphi'(0) > 0`` ⟹ `LINESEARCH_NO_DESCENT`;
- ``\varphi'(0) = 0`` ⟹ `LINESEARCH_STATIONARY`, which for the ``\|F\|^2`` merit of a
  [`NonlinearSolver`](@ref) *is* the exact root.

Either way the caller's trial step is handed back — never the ``\alpha = 0`` anchor, which would
freeze the outer iterate.

### The smallest informative step ``\alpha_\mathrm{min}``

For the shrinking ladder of [`Backtracking`](@ref) there is a floor below which the sufficient
decrease condition could only be decided by rounding, namely where
``c_1\alpha|\varphi'(0)|`` falls below an ulp of ``\varphi(0)``. That step is
[`SimpleSolvers.backtracking_αmin`](@ref), and the ladder stops there rather than at `eps`. It is
reported in the `αmin` field of the [`LinesearchStatus`](@ref) and is `zero` — meaning *not
applicable* — for the bracketing and minimising searches, which do not shrink.

### The contract every method keeps

Reaching a method through [`solve`](@ref) or [`solve_with_status`](@ref) guarantees six things; see
[`LinesearchMethod`](@ref) for the full statement:

1. **it never throws** — a situation it cannot handle is reported, not raised, the one exception
   being a `params.αmax` that is not a usable ceiling, which is a caller error and raises before
   any evaluation;
2. **it returns ``\alpha > 0``** — never the anchor, never a negative step;
3. **it reports through [`SimpleSolvers.linesearch_warnings`](@ref) only**, and only when the
   *user* called it;
4. **a non-finite or ascending anchor is reported, not assumed away**;
5. **cost is bounded independently of the merit's scale** — multiplying ``\varphi`` by a constant
   must not change the number of evaluations;
6. **it returns ``\alpha \leq \alpha_\mathrm{max}``** — see [Bounding the
   step](@ref "Bounding the step").

Clause 3 is the one that decides where the criteria above are *visible*. `solve` runs the search
and then reports; `solve_with_status` runs it and stays silent. Genuine failures
(`EXHAUSTED`, `NO_DESCENT`) are reported at `verbosity ≥ 1` and the two benign-but-notable ones
(`FLOOR`, `STATIONARY`) at `verbosity ≥ 2`, because warning about a round-off floor at the default
verbosity means warning about success.

## Part 2 — the nonlinear solver

### The three residuals

[`SimpleSolvers.residuals`](@ref) measures, once per iteration:

| symbol | meaning |
|---|---|
| ``r^x_s`` | successive residual in ``x``, ``\|x - \bar{x}\|`` |
| ``r^f_a`` | absolute residual, ``\|F(x)\|`` |
| ``r^f_s`` | successive residual in ``F``, ``\|F(x) - F(\bar{x})\|`` |

### The residual gate

Everything below turns on one predicate, [`SimpleSolvers.residual_small`](@ref):

```math
r^f_a \leq \texttt{f\_abstol} + \texttt{f\_reltol}\cdot\|F(x_0)\| .
```

It is required by both convergence branches and **negated** by both give-up branches, which is what
makes converging and stagnating mutually exclusive by construction rather than by agreement between
two tests.

!!! warning "The default `f_abstol` is zero and `f_reltol` tightens with a good guess"
    With the default `f_abstol = 0` the absolute branch is switched *off*: convergence is decided
    entirely by the relative and successive branches. And the relative term is anchored at the
    *initial* residual ``\|F(x_0)\|``, so a better initial guess makes the gate **tighter**, not
    looser. An `f_abstol` below the round-off floor of your own residual is unsatisfiable, and such
    a solve is reported as stagnated rather than converged. See [`Options`](@ref).

### Convergence

[`SimpleSolvers.assess_convergence`](@ref) returns four flags:

- `x_converged` ⟺ [`SimpleSolvers.iterate_settled`](@ref) (``r^x_s \leq \|x\|\cdot`` `x_suctol`)
  **and** `residual_small`;
- `f_converged` ⟺ (``r^f_s \leq \|F(x)\|\cdot`` `f_suctol` **and** `residual_small`) **or**
  ``r^f_a \leq`` `f_abstol`;
- `f_increased` ⟺ ``\|F(x)\| > \|F(\bar{x})\|``;
- `stalled` ⟺ `iterate_settled` **and not** `residual_small`.

[`SimpleSolvers.isconverged`](@ref) is `x_converged || f_converged`. The `residual_small`
conjunct on the successive branches is not decoration: a step that stalls makes ``r^x_s`` and
``r^f_s`` vanish while ``r^f_a`` is still large, and without the gate that would be reported as
convergence.

### Divergence, stagnation and giving up

Four distinct ways a solve ends without converging:

| predicate | what it detects | controlled by |
|---|---|---|
| [`SimpleSolvers.stalled_step`](@ref) | the iterate **froze** while the residual is not small | `max_stalls` (default 2) |
| [`SimpleSolvers.no_progress`](@ref) | the iterate **moves** but the residual is going nowhere | `f_stall_window` (default 0, **off**) |
| [`SimpleSolvers.havenonfinite`](@ref) | any residual is not finite — the iteration left the representable region | always on, from iteration 1 |
| `f_increased` | the residual grew | `allow_f_increases` (default `true`, so off) |

The first two cover disjoint cases and their thresholds differ by orders of magnitude for a reason.
Two consecutive stalled steps are conclusive because the second one ran with a *freshly evaluated*
Jacobian (a stall forces one — see below), whereas no number of *moving* steps is conclusive about
a convergence rate: an iteration converging linearly with rate ``\rho`` improves by ``\rho^W`` over
a window ``W``, so a window of 50 at the default factor abandons every ``\rho > 0.986``, which a
[`PicardSolver`](@ref) on a stiff problem is slower than. That is why stalling has a default and
lack of progress is opt-in — the *diagnosis* of the latter is unconditional (it is consulted only
about a solve that has already spent its budget) while the *stopping* is the caller's to ask for.

A non-finite direction is not on that list because it is not a stopping criterion at all: it is
rejected outright by [`SimpleSolvers.solver_step!`](@ref) as a `NonlinearSolverException`. It
cannot be recovered from — [`SimpleSolvers.nan_recovery!`](@ref) damps by a factor, and
`Inf * nan_factor` is `Inf` — whereas a non-finite trial *residual* is damped.

### The full stopping test

[`SimpleSolvers.meets_stopping_criteria`](@ref) is the disjunction, tested **before the first step**
as well as after each one:

- `isconverged` and `iterations ≥ min_iterations`;
- [`SimpleSolvers.isstalled`](@ref) and `iterations ≥ min_iterations`;
- [`SimpleSolvers.isnotprogressing`](@ref) and `iterations ≥ min_iterations`;
- `f_increased` and `!allow_f_increases`;
- `iterations ≥ max_iterations`;
- ``r^f_a >`` `f_abstol_break` (default `Inf`);
- `havenonfinite` and `iterations ≥ 1`.

### What is reported, and when

[`SimpleSolvers.nonlinear_solver_warnings`](@ref) fires **once per solve**, at the end. Its three
"this solve did not do what you asked" messages are mutually exclusive, most specific first:
stagnation (the iterate froze) replaces lack of progress (the iterate moves, the residual does
not), which replaces the bare iteration count. They are rate-limited by
[`SimpleSolvers.should_report!`](@ref), which reports the 1st, 2nd, 4th, 8th … occurrence of a
diagnosis — so a caller looping `solve!` per time step over an unattainable tolerance is not
flooded, while a solve that starts failing for a *new* reason is reported at once.

## How a line search's verdict reaches the solver

This is the connection between the two halves, and it runs entirely through data — a line search
logs nothing from inside a solve.

Within one [`SimpleSolvers.solver_step!`](@ref):

1. the solver calls `solve_with_status(linesearch(s), one(T), lsparams)`, passing
   `lsparams = (x = x, parameters = params, φ₀ = ‖F(x)‖²)`. The `φ₀` field is the merit at the
   anchor, which the solver has *already* computed, so the line search does not re-evaluate it;
2. the outcome is counted into a per-solve tally by
   [`SimpleSolvers.record_linesearch!`](@ref);
3. if the outcome is `LINESEARCH_FLOOR` or `LINESEARCH_NO_DESCENT`, the step is flagged with
   [`SimpleSolvers.flag_stall!`](@ref) — the line search *knows* the iteration cannot progress
   along this direction, one iteration before the step-based [`SimpleSolvers.stalled_step`](@ref)
   would see it;
4. on `LINESEARCH_NO_DESCENT` the step is **not taken**: moving along a direction that cannot
   decrease the merit would only make the forced retry start from a worse point. The line search
   still returned ``\alpha > 0``, as its contract requires — whether to use it is the caller's
   decision, and here the caller declines.

The flag then has two consumers:

- [`SimpleSolvers.needs_refresh`](@ref) is true for the next step, which
  [`SimpleSolvers.solver_step!`](@ref) passes to [`SimpleSolvers.maybe_refactorize!`](@ref) as
  its `stalled` keyword. A quasi-Newton solver therefore rebuilds its [`Jacobian`](@ref)
  *immediately* rather than waiting for the next `refactorize` multiple — which is what makes
  `max_stalls = 2` conclusive for every `refactorize` and not only for `refactorize = 1`;
- [`SimpleSolvers.record_stall!`](@ref) consumes it once per iteration, OR-ed with
  `iterate_settled` and gated by the *same* `!residual_small` as everything else:

  ```julia
  stalled = (flagged || iterate_settled(rxₛ, config, state)) && !residual_small(rfₐ, config, state)
  ```

  That gate is why a converged solve — whose last line search reports the floor as a matter of
  course — is not counted as stagnating.

The tally is copied into the [`NonlinearSolverStatus`](@ref) and read three ways:

- [`SimpleSolvers.linesearch_outcomes`](@ref) — the raw counts, indexed by
  [`SimpleSolvers.linesearch_index`](@ref);
- [`SimpleSolvers.linesearch_failures`](@ref) — how many steps reported a non-benign outcome;
- [`SimpleSolvers.dominant_linesearch_outcome`](@ref) — the one worth naming, which
  [`SimpleSolvers.linesearch_reason`](@ref) appends to the solver's own failure message so that a
  failed solve names the *cause* and not only the symptom ("the line search reported
  `LINESEARCH_NO_DESCENT` on 194 of the 200 step(s)").

!!! info "Read the tally, do not scrape the log"
    A program acting on a rejected line search — restarting an approximate Hessian, falling back to
    steepest descent — should use [`solve_with_status!`](@ref) and read the tally. The log is for
    the user. This is the whole reason [`solve_with_status`](@ref) is the method extension point
    and [`solve`](@ref) is derived from it: there is no path by which the package reaches a
    method's `solve` during a solve, so the guarantee holds by construction.

## The options that participate

| option | default (`Float64`) | governs |
|---|---|---|
| `f_abstol` | `0` | the absolute branch of `residual_small` — **off** by default |
| `f_reltol` | `√eps` | the relative branch, against ``\|F(x_0)\|`` |
| `x_suctol` | `2eps` | `iterate_settled`, hence `x_converged` and `stalled_step` |
| `f_suctol` | `2eps` | the successive branch of `f_converged` |
| `max_stalls` | `2` | consecutive stalled steps before giving up |
| `f_stall_window` | `0` (**off**) | iterations without progress before giving up |
| `f_stall_factor` | `0.5` | the residual drop that counts as progress |
| `f_abstol_break` | `Inf` | a residual so large the solve is abandoned |
| `allow_f_increases` | `true` | whether a growing residual stops the solve |
| `min_iterations` | `0` | floor below which convergence and the two give-up tests are ignored |
| `max_iterations` | `1000` | the **outer** iteration budget |
| `linesearch_max_iterations` | `60` | the **inner**, one-dimensional budget |
| `verbosity` | `1` | `0` silences everything; `2` adds the benign line-search outcomes |

`max_iterations` and `linesearch_max_iterations` are deliberately distinct: a one-dimensional
search inside a single solver step needs a budget on the order of the mantissa width
([`SimpleSolvers.linesearch_iterations`](@ref)), not thousands of trials.
