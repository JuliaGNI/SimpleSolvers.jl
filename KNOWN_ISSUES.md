# Known issues

## Reported by GeometricOptimizers.jl, not addressed in 0.12.0

Of their reports, **D3**, **D4** and **D6** are addressed in 0.12.0 and are written up there; what
follows is what is left.

### K1 · `store_trace`, `show_trace` and `extended_trace` have no readers.

- **location:** `src/`
- **evidence:** `grep` over `src/` finds
  them only in `Options` itself — they are accepted, printed by `show(::Options)`, and then
  ignored, so a caller who sets one gets silence rather than a trace or an error.
  GeometricOptimizers implements `store_trace` itself for this reason. Either implement them or
  remove them; both are breaking, so neither belongs in a release driven by something else.
- **kind:** dead code
- **found:** 2026-08-14; one issue with GeometricOptimizers `D2` and `B2`

## Raised while fixing D6 (the step ceiling), not addressed

### K2 · The default ceiling does not close A1b, and cannot.

- **location:** —
- **evidence:** `DEFAULT_LINESEARCH_αmax = 65536` removes
  the unbounded extrapolation and bounds the bracketing cost, but at `‖δ‖ ≈ 5.5` it still permits
  `‖αδ‖ = 3.6e5`, five orders above the `2π` past which retracting a lift only adds round-off. Four
  of the eight SVD starting points still diverge under `Cayley` with either polynomial search; the
  same runs at `αmax = 1` all converge (the table in the 0.12.0 entry). The remaining half is
  GeometricOptimizers passing `params.αmax = c·2π/‖δ‖`, which is theirs to write — and which they
  cannot pick up until their `[compat]` moves off `SimpleSolvers = "0.11"`. **A1b stays open until
  then.** Choosing the constant `c` is a decision about their geometry, not about `φ`, which is
  exactly why this package does not make it for them.
- **kind:** defect
- **found:** 2026-08-14

### K3 · A binding ceiling leaves no trace in the `LinesearchStatus`.

- **location:** —
- **evidence:** By design there is no
  `LINESEARCH_CAPPED` (see the 0.12.0 entry for why), and a caller who set `params.αmax` can
  compare it against `steplength`. A caller relying on the *method's* ceiling cannot: "the
  minimiser is at 65536" and "the search stopped because it was not allowed past 65536" are
  reported identically, both as `LINESEARCH_DECREASED`. A boolean field on the status would close
  it without touching the outcome enum or its tally; left out because nothing needed it yet and the
  struct is copied per solver step.
- **kind:** defect
- **found:** 2026-08-14

### K4 · The capped path costs one merit evaluation that was already made.

- **location:** —
- **evidence:** `BierlaireQuadratic` and
  `Bisection` reach `capped_status`, which evaluates `φ(αmax)` — the same point the bracketing
  evaluated on the round it stopped. Carrying the value out of `_triple_point_core` and
  `_bracket_core` (the fixed-point bracketer already returns its endpoint values) would remove it.
  One evaluation, on a path that is rare in a Euclidean problem and is a full residual or objective
  evaluation when it fires, so it is worth doing and was not urgent. The *second* duplicate on that
  path — the one `_bracket_core` made by re-probing a ceiling `_bracket_minimum_core` had already
  reached — is gone.
- **kind:** defect
- **found:** 2026-08-14

### K5 · A ceiling that binds can be reported as `LINESEARCH_FLOOR`.

- **location:** —
- **evidence:** `capped_status` classifies the
  step at `αmax` by the same `τ` rule as any other returned step, so a merit that is still falling
  at the ceiling — which is what `:capped` *means* — but has fallen by less than `τ` over the whole
  admissible range comes back as a floor. That is a claim about the **direction**, that no line
  search can progress along it, and the outer iteration acts on it through `flag_stall!` and
  `max_stalls`; what was actually established is only that no step the caller *permits* decreases
  the merit measurably. It is the same shape as the two unearned floors 0.12.0 removed from
  `Bisection`, reached through a third door, and it is reachable exactly where the caller's ceiling
  is tightest — the case the ceiling exists for. Left standing because the alternative is not
  obviously better: `LINESEARCH_EXHAUSTED` says "no step was found", which is false of a step that
  was found and returned, and a caller that supplied the ceiling can compare it against
  `steplength`. Closing it properly means the boolean-on-the-status of the entry above, not a
  different outcome.
- **kind:** defect
- **found:** 2026-08-15; one issue with GeometricOptimizers `D7`

### K6 · The public bracketers can return a degenerate interval at the ceiling.

- **location:** —
- **evidence:** When `αmax` lies at or
  below the bracketing start, `_bracket_core` returns `(a, a, :capped)` and `bracket` /
  `bracket_minimum` hand that on as the interval `(a, a)` with no signal. Unreachable from a line
  search — `Quadratic` and `BierlaireQuadratic` return the ceiling instead of bracketing when their
  start is not strictly below it, and `Bisection` always brackets from `α = 0`, which every valid
  ceiling is strictly above — but reachable by a standalone caller of the exported
  `bracket_minimum`, which has no such guard.
- **kind:** defect
- **found:** 2026-08-14

### K7 · `:unbracketable` is now nearly unreachable, and with it the diagnosis it carried.

- **location:** —
- **evidence:** A rightward
  search under a finite ceiling always terminates as `:ok` or `:capped`, so `LINESEARCH_EXHAUSTED`
  from a *failed bracket* now needs `αmax = Inf` or a flipped, leftward search. That is the right
  behaviour for the case it was measured on — a merit that descends forever is better answered with
  the largest admissible step than with a failure — but it means a genuinely unbracketable merit is
  now reported as an ordinary decrease at the ceiling. This is the same gap as the previous entry
  seen from the other side: the status cannot say "I stopped because you would not let me look
  further".
- **kind:** defect
- **found:** 2026-08-14

### K8 · `StrongWolfe` classifies the ceiling differently from the minimising searches.

- **location:** —
- **evidence:** On a merit
  whose minimiser lies beyond it, the three minimising searches return `αmax` with
  `LINESEARCH_DECREASED` while `StrongWolfe` returns `αmax` with `LINESEARCH_EXHAUSTED` (observed
  on `φ(α) = (α - 10^7)²/10^{14}`). Its behaviour is unchanged and internally consistent — the
  strong curvature condition genuinely was never met, and it has always reported the last
  Armijo-acceptable step that way — but the ceiling makes the divergence between the two families
  visible where it previously took a contrived merit to reach. Whether a Wolfe search *should* call
  a step that was never allowed to grow "exhausted" is a question about that method, not about the
  ceiling, so it was left alone.
- **kind:** defect
- **found:** 2026-08-14

## Raised while fixing the `Bisection` maximum, not addressed

### K9 · The orientation is not checked on a bracket that `bracket_minimum` flipped.

- **location:** —
- **evidence:**
  `_bisect_for_minimum` returns early on `lo ≤ 0`, which covers two different things: the overshoot
  branch, where `lo = 0` and `ylo` *is* the `φ′(0) < 0` the anchor check established (genuinely
  nothing to check), and a bracket that the walk flipped leftward, where `lo < 0` and nothing has
  been established at all. The second is a real hole, and it is demonstrable: on `φ(α) = α²` with
  `φ′` given the roots `-0.5`, `0.3`, `0.7`, `bracket_minimum` returns `[-1, 1]` with `φ′(-1) > 0`,
  and the bisection converges to `α = -0.5` — a **maximum**, reached with the check skipped.

  Nothing leaked in that instance, because `-0.5 < 0` and the α > 0 contract rejected it: the
  negative-step retry fired and the search reported `LINESEARCH_EXHAUSTED`. Whether the same route
  can select a maximum at *positive* α — the interval spans zero and may contain several `+ → -`
  crossings, only some of them negative — was **not** established either way. It is hard to arrange
  because `check_anchor` guarantees `φ′(0) < 0`, so a crossing selected from a positive left
  endpoint has one candidate at or before zero; it is not ruled out. The clean closure is to bisect
  `[0, hi]` rather than `[lo, hi]` whenever `lo < 0` — a positive step is all the contract permits
  anyway, and `φ′(0) < 0` orients it by construction — but that removes the flipped-bracket case the
  negative-step retry exists to handle, so it wants its own change and its own tests.
- **kind:** defect
- **found:** 2026-08-15

### K10 · The repair throws away the first bisection.

- **location:** —
- **evidence:** When `ylo > 0`, `_bisect_for_minimum` has already
  run a complete bisection to a root it then discards, and pays for a second one. Checking the
  orientation *before* the loop — an argument to `_bisection_core` saying which sign the left
  endpoint must have, or a two-evaluation pre-flight — would spend two derivative evaluations
  instead of a whole bisection. Left because the path is the pathological one and the cost on the
  common path, which is what the fix was careful about, is already zero.
- **kind:** defect
- **found:** 2026-08-15

### K11 · `_bisection_core` decides its root test with `f_abstol`, which is a residual tolerance.

- **location:** —
- **evidence:** The
  loop stops early on `≈(y, 0; atol = config.f_abstol)` where `y` is `φ′(α)`, while `Options`
  documents `f_abstol` as *"an absolute target for ‖F(x)‖"*. Those are incommensurable: `φ′` is the
  α-derivative of `‖F‖²`, not a residual norm. It is invisible at the default `f_abstol = 0`, where
  the branch fires only on an exact zero and the loop terminates on its width test instead — but it
  is not invisible when a caller sets one. Measured on `φ(α) = (α-1)²`: `f_abstol = 0` locates the
  minimiser to `|φ′| = 8.9e-16`, `1e-6` to `9.5e-7`, and `1e-2` to `7.8e-3`, i.e. raising the
  residual tolerance for a coarse solve silently coarsens the line minimiser by the same factor.
  This is the same class of defect as the one 0.10.0 fixed in `BierlaireQuadratic`, where "one
  absolute constant used to govern three incommensurable quantities"; the fix there was to compare
  merit differences against `τ` and α-space widths against `ε`. The equivalent here is a tolerance
  of the bisection's own, defaulting to something derived from `φ′(0)`. Left out because it changes
  the accuracy of every `Bisection` solve that sets `f_abstol`, which is a behaviour change with no
  reported symptom behind it.
- **kind:** defect
- **found:** 2026-08-15

## Documentation

### K12 · There is no page for the nonlinear solvers.

- **location:** `docs/src/convergence.md`
- **evidence:** 0.12.0 adds `docs/src/convergence.md`, which
  documents every line-search outcome, every solver stopping criterion, and the channel between
  them — but `NewtonSolver`, `PicardSolver` and their caches, states and constructors are still
  reachable only through docstrings and the `@autodocs` index. The line searches have nine pages
  and the solvers that call them have none.
- **kind:** docs
- **found:** 2026-08-14

## Introduced or left standing by 0.12.0

### K13 · `should_report!` does not promise every occurrence, and its counters are global.

- **location:** —
- **evidence:** A repeating
  diagnosis is reported on occurrences 1, 2, 4, 8, 16 … — occurrence 10 is silent. This is the
  intended trade (see its docstring) and not a defect, but it *is* a behaviour a caller can be
  surprised by, and there is no `Options` knob to opt out of it: the only controls are
  `verbosity = 0` and `reset_warning_counts!`. The counters are also process-global and shared
  across unrelated solvers, which is deliberate — a per-solver counter would reset on every step of
  a loop that rebuilds its solver, which is exactly the flood the cap exists for — but it means two
  independent solves can suppress each other's reports.
- **kind:** defect
- **found:** 2026-08-14
