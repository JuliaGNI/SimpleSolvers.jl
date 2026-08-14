@doc raw"""
    LinesearchOutcome

Why a [`LinesearchMethod`](@ref) stopped. Stored in a [`LinesearchStatus`](@ref), which is
returned by [`solve_with_status`](@ref).

- `LINESEARCH_DECREASED`: a step was found that decreased the merit by more than the round-off
  allowance ``\tau``. This is the only outcome that reports progress. Note what it does *not*
  claim: [`Backtracking`](@ref) and [`StrongWolfe`](@ref) additionally verify their Wolfe
  condition before returning, whereas the minimising searches ([`Bisection`](@ref),
  [`Quadratic`](@ref), [`BierlaireQuadratic`](@ref)) approximate the line minimiser and never
  test one. The common guarantee across all of them is the ``\tau``-exceeding decrease.
- `LINESEARCH_FLOOR`: the merit has reached its round-off floor — no trial step changes it by
  more than ``\tau``, so *no* line search can make progress here. The returned step is the
  smallest informative one. This is not an error and is only reported at `verbosity ≥ 2`: it is
  the *expected* final state of a converged solve, and when it is *not* — when the residual is
  still large — the outer iteration reports it as stagnation instead (see
  [`stalled_step`](@ref) and [`Options`](@ref)).
- `LINESEARCH_EXHAUSTED`: no acceptable step although the merit *does* vary by more than
  ``\tau``. Either ``\varphi'(0)`` is inconsistent with ``\varphi`` (a stale or regularized
  [`Jacobian`](@ref), an inexact linear solve, a non-smooth merit), or the
  `linesearch_max_iterations` budget of [`Options`](@ref) was spent.
- `LINESEARCH_NO_DESCENT`: ``\varphi'(0) > 0``, or ``\varphi(0)``/``\varphi'(0)`` is not
  finite. No ``\alpha`` can satisfy the sufficient decrease condition.
- `LINESEARCH_STATIONARY`: ``\varphi'(0) = 0``, e.g. a vanishing direction at an exact root.
  Benign — there is nothing to search for.
- `LINESEARCH_UNKNOWN`: the method does not report an outcome — [`Static`](@ref), which
  evaluates no merit and so has established nothing, and any third-party method that chooses not
  to report one.
"""
@enum LinesearchOutcome::Int8 begin
    LINESEARCH_DECREASED
    LINESEARCH_FLOOR
    LINESEARCH_EXHAUSTED
    LINESEARCH_NO_DESCENT
    LINESEARCH_STATIONARY
    LINESEARCH_UNKNOWN
end

"""
    const NLINESEARCH_OUTCOMES

The number of [`LinesearchOutcome`](@ref)s, i.e. the length of the tally a
[`NonlinearSolverState`](@ref) keeps of the outcomes its line search reported (see
[`record_linesearch!`](@ref)). A `const` rather than `length(instances(LinesearchOutcome))` at
each use, because it is a type parameter of that tally's `MVector` and so has to be known to the
compiler.
"""
const NLINESEARCH_OUTCOMES = length(instances(LinesearchOutcome))

"""
    linesearch_index(oc)

The index of the [`LinesearchOutcome`](@ref) `oc` in a tally of length
[`NLINESEARCH_OUTCOMES`](@ref) — the enum's value plus one, since the enum starts at zero and
Julia indexes from one. See [`record_linesearch!`](@ref) and
[`linesearch_outcomes`](@ref).
"""
linesearch_index(oc::LinesearchOutcome) = Int(oc) + 1

"""
    isbenign(oc)

`true` for the [`LinesearchOutcome`](@ref)s that report no failure: `LINESEARCH_DECREASED` (a
genuine decrease), `LINESEARCH_STATIONARY` (nothing to search for) and `LINESEARCH_UNKNOWN` (the
method does not report one). The remaining three — `LINESEARCH_FLOOR`, `LINESEARCH_EXHAUSTED`
and `LINESEARCH_NO_DESCENT` — are what [`linesearch_failures`](@ref) counts and what
[`linesearch_warnings`](@ref) reports on.

`LINESEARCH_FLOOR` counts as a failure here even though it is the *expected* final state of a
converged solve, because whether it matters is the outer iteration's call and this is how the
outer iteration is told: the tally is read by [`nonlinear_solver_warnings`](@ref), which names it
only for a solve that did *not* converge.
"""
isbenign(oc::LinesearchOutcome) =
    oc === LINESEARCH_DECREASED || oc === LINESEARCH_STATIONARY || oc === LINESEARCH_UNKNOWN

@doc raw"""
    LinesearchStatus{T}

The step length returned by a line search together with the diagnostics needed to tell
*progress* from *stagnation*. Obtained from [`solve_with_status`](@ref); compare this to
[`NonlinearSolverStatus`](@ref), which plays the same role for the outer iteration.

The step length alone cannot express the difference: a tiny ``\alpha`` may be the correct
answer, or it may be all that is left after the merit turned out to be irreducible. See
[`LinesearchOutcome`](@ref).

# Keys

- `α`: the returned step length (the value [`solve`](@ref) returns),
- `outcome::`[`LinesearchOutcome`](@ref),
- `trials`: what the search cost, as the number of times it evaluated the problem — *not* the
  `linesearch_max_iterations` budget. That is the merit for every method except
  [`Bisection`](@ref), which drives on the derivative it bisects and brackets on the merit, so
  its count is of both. For [`Backtracking`](@ref) and [`StrongWolfe`](@ref) it is exactly the
  number of trial steps ``\alpha > 0``: every merit evaluation is either the ``\alpha = 0``
  anchor or a counted trial. For the searches that *bracket* it includes what the bracketing
  spent ([`bracket_minimum`](@ref), [`triple_point_finder`](@ref)) — that is where those searches
  do their work, and on the path where a ceiling binds it is the whole of it, so a count omitting
  it reported one evaluation, or none at all, for a search of any size. One of the evaluations it
  then counts is the bracketing's own re-evaluation of the anchor, so for those methods the
  number is the cost rather than exactly the number of distinct positive steps,
- `φ₀`, `d₀`: the merit and its derivative at the anchor ``\alpha = 0``,
- `φ`: the merit at the returned step,
- `τ`: the round-off resolution of the merit (see [`armijo_tolerance`](@ref)), against which
  every method decides whether the decrease it achieved was genuine,
- `αmin`: the smallest step length that could still be decided by the merit rather than by
  rounding (see [`backtracking_αmin`](@ref)). This is a *shrinking-ladder* quantity and is
  therefore `zero` — meaning "not applicable" — for the minimising searches
  ([`Bisection`](@ref), [`Quadratic`](@ref), [`BierlaireQuadratic`](@ref)) and for
  [`StrongWolfe`](@ref), which bracket rather than shrink.
"""
struct LinesearchStatus{T}
    α::T
    outcome::LinesearchOutcome
    trials::Int

    φ₀::T
    d₀::T
    φ::T
    τ::T
    αmin::T
end

"""
    LinesearchStatus(α, outcome=LINESEARCH_UNKNOWN)

Construct a [`LinesearchStatus`](@ref) that carries only the step length and the
[`LinesearchOutcome`](@ref); the remaining diagnostics are filled with `NaN`/zero. For a method
that does not measure them — [`Static`](@ref), or a third-party method whose
[`solve_with_status`](@ref) has nothing more to say.
"""
LinesearchStatus(α::T, outcome::LinesearchOutcome=LINESEARCH_UNKNOWN) where {T} =
    LinesearchStatus{T}(α, outcome, 0, T(NaN), T(NaN), T(NaN), zero(T), zero(T))

"""
    steplength(status)

The step length stored in `status::`[`LinesearchStatus`](@ref), i.e. what [`solve`](@ref)
returns.
"""
steplength(status::LinesearchStatus) = status.α

"""
    outcome(status)

The [`LinesearchOutcome`](@ref) stored in `status::`[`LinesearchStatus`](@ref).
"""
outcome(status::LinesearchStatus) = status.outcome

"""
    trials(status)

The number of trial steps at which the merit was evaluated, stored in
`status::`[`LinesearchStatus`](@ref).
"""
trials(status::LinesearchStatus) = status.trials

"""
    issufficient(status)

`true` if the line search found a step with a *genuine* sufficient decrease, i.e. one that
decreased the merit by more than the round-off allowance `τ` of `status`. Compare
[`isfloor`](@ref).
"""
issufficient(status::LinesearchStatus) = outcome(status) === LINESEARCH_DECREASED

"""
    isfloor(status)

`true` if the line search could not find *any* step that changes the merit by more than the
round-off allowance `τ`, i.e. the merit has reached its round-off floor. The outer iteration
cannot make progress in this state no matter how the step is chosen — which is why a
[`NonlinearSolver`](@ref) counts it as a stalled step (see [`record_stall!`](@ref)).
"""
isfloor(status::LinesearchStatus) = outcome(status) === LINESEARCH_FLOOR

Base.show(io::IO, s::LinesearchStatus) = print(io,
    "LinesearchStatus: α = $(s.α), $(s.outcome) after $(s.trials) trial step(s) ",
    "(φ(0) = $(s.φ₀), φ(α) = $(s.φ), φ'(0) = $(s.d₀), τ = $(s.τ), αmin = $(s.αmin)).")

@doc raw"""
    check_anchor(φ₀, d₀, α, αmax=Inf)

Validate the ``\alpha = 0`` anchor of a line search problem. Return a
[`LinesearchStatus`](@ref) that the caller should return *immediately*, or `nothing` if the
anchor is usable and the search may proceed.

This is the one definition of the anchor policy shared by every [`LinesearchMethod`](@ref):

- ``\varphi(0)`` or ``\varphi'(0)`` not finite, or ``\varphi'(0) > 0``, gives
  `LINESEARCH_NO_DESCENT`: no ``\alpha`` can decrease the merit along this direction, so
  shrinking or bracketing would only spend merit evaluations to discover that. The caller's
  trial step `α` is handed back — never the ``\alpha = 0`` anchor, which would freeze the outer
  iterate (`x .+= 0 .* d`).
- ``\varphi'(0) = 0`` gives `LINESEARCH_STATIONARY`. For the ``\|F\|^2`` merit of a
  [`NonlinearSolver`](@ref) this *is* the exact root (``F = 0 \Rightarrow \varphi'(0) = 0`` and
  the direction vanishes), so it is benign and every ``\alpha`` is equivalent.

An ascent anchor arises in practice when the direction did not come from an exact, freshly
factorized Newton solve — a stale [`Jacobian`](@ref) under `refactorize > 1`, a nonzero
`regularization_factor`, or an inexact linear solve. The correct response is to refresh the
Jacobian, which is why the line search reports the situation instead of trying to salvage a step
from it, and [`solver_step!`](@ref) acts on the report: on `LINESEARCH_NO_DESCENT` it leaves the
iterate where it is (moving along a direction that cannot decrease the merit would only make the
retry start from a worse point) and records a stall, which forces a fresh Jacobian on the next
step (see [`needs_refresh`](@ref) and [`maybe_refactorize!`](@ref)) and gives up after
`max_stalls` if that does not help.

The step handed back is therefore still positive, as the contract requires — whether to *use*
it is the caller's decision, not the line search's. It also respects `αmax` (see
[`linesearch_αmax`](@ref)), so that the ceiling holds on *every* return of a line search and not
only on the ones that searched.
"""
function check_anchor(φ₀::T, d₀::T, α::T, αmax::T=T(Inf)) where {T}
    # A non-positive trial step is not a step the caller can be handed back, so substitute the
    # unit step — the same convention `StrongWolfe` uses for its initial trial. This is what
    # makes the α > 0 guarantee hold even when the caller passes α ≤ 0. The substitute is bounded
    # by the ceiling like any other step, which matters when a caller both passes α ≤ 0 and asks
    # for an αmax below one.
    αout = min(α > zero(T) ? α : one(T), αmax)
    if !isfinite(φ₀) || !isfinite(d₀) || d₀ > zero(T)
        LinesearchStatus{T}(αout, LINESEARCH_NO_DESCENT, 0, φ₀, d₀, φ₀, zero(T), zero(T))
    elseif iszero(d₀)
        LinesearchStatus{T}(αout, LINESEARCH_STATIONARY, 0, φ₀, d₀, φ₀, zero(T), zero(T))
    end
end

@doc raw"""
    capped_status(prob, params, αmax, φ₀, d₀, τ, n=0)

The [`LinesearchStatus`](@ref) of a search whose bracketing reached the ceiling `αmax` (see
[`linesearch_αmax`](@ref)) with the merit still falling. The turning point then lies beyond the
largest step the caller allows, so `αmax` *is* the best admissible step, and this is the shared
definition of what to report about it.

It is deliberately not a failure and not a distinct [`LinesearchOutcome`](@ref). The merit is
evaluated at `αmax` and classified by exactly the rule every other returned step is classified by:
`LINESEARCH_DECREASED` when it beats ``\varphi(0)`` by more than the round-off allowance ``\tau``,
`LINESEARCH_FLOOR` when it does not. That is honest in both directions — on a compact merit, where
this case arises, ``\varphi(\alpha_\mathrm{max})`` is genuinely lower and the step genuinely
decreases the merit — and a caller that wants to know whether its ceiling bound the search can
compare the ceiling it supplied against [`steplength`](@ref).

`n` is what the bracketing that reached the ceiling spent, which the caller gets from the
bracketing core (see [`_bracket_core`](@ref)). It has to be passed in rather than assumed, because
on this path the bracketing *is* the search: reporting only the single evaluation made here would
make [`trials`](@ref) say a capped search cost one step whatever it actually cost.
"""
function capped_status(prob, params, αmax::T, φ₀::T, d₀::T, τ::T, n::Int=0) where {T}
    φ = value(prob, αmax, params)
    LinesearchStatus{T}(αmax, φ ≤ φ₀ - τ ? LINESEARCH_DECREASED : LINESEARCH_FLOOR,
        n + 1, φ₀, d₀, φ, τ, zero(T))
end

@doc raw"""
    solve_with_status(ls, α, params=NullParameters())

Like [`solve`](@ref), but return a [`LinesearchStatus`](@ref) — the step length *plus* the
reason the search stopped — and emit no log messages. Use [`linesearch_warnings`](@ref) to
report the status; that is all [`solve`](@ref) adds.

This is what a *program* calls, and it is the [`LinesearchMethod`](@ref) extension point: every
built-in method (all six of [`Backtracking`](@ref), [`StrongWolfe`](@ref), [`Bisection`](@ref),
[`Quadratic`](@ref), [`BierlaireQuadratic`](@ref) and [`Static`](@ref)) implements *this*, and
gets [`solve`](@ref) derived from it. A method that reports no outcome of its own returns
`LINESEARCH_UNKNOWN`, as [`Static`](@ref) does.

!!! warning "A method must implement this"
    There is no fallback: the generic method below raises rather than deriving a status from
    [`solve`](@ref). It used to do exactly that, and the derivation ran the wrong way — a method
    that defined only `solve` was then reached *through* `solve` from inside every iteration of a
    [`NonlinearSolver`](@ref), and emitted its messages there, which is the one thing the contract
    in [`LinesearchMethod`](@ref) promises does not happen. Deriving `solve` from this instead
    makes that promise structural. A third-party method that defines only `solve` therefore has to
    move its body here; the boilerplate it used to carry (`solve_with_status`, then
    [`linesearch_warnings`](@ref), then [`steplength`](@ref)) is what it gets for free in exchange.
"""
function solve_with_status(ls::Linesearch{T}, α::T, params=NullParameters()) where {T}
    throw(ArgumentError("$(nameof(typeof(method(ls)))) does not implement `solve_with_status`, " *
                        "which is what a LinesearchMethod implements; `solve` is derived from it. " *
                        "Define `solve_with_status(::Linesearch{T,<:$(nameof(typeof(method(ls))))}, α::T, params)` " *
                        "returning a LinesearchStatus (use `LinesearchStatus(α, LINESEARCH_UNKNOWN)` " *
                        "if the method has no outcome to report)."))
end

"""
    curvature_diagnostic(status, ls, params)

Method-specific extra diagnostic emitted by [`linesearch_warnings`](@ref) at
`verbosity ≥ 2`. The fallback does nothing; [`Backtracking`](@ref) checks the
[`CurvatureCondition`](@ref), which costs a derivative evaluation — a full
[`Jacobian`](@ref) for the line search problem of a [`NonlinearSolver`](@ref), hence the
verbosity gate.

Reached only from a direct [`solve`](@ref) call, since that is the only caller of
[`linesearch_warnings`](@ref). A `NonlinearSolver` at `verbosity = 2` therefore no longer pays for
this once per iteration; to see it for a step of a solve, call the line search on that step's
problem directly.
"""
curvature_diagnostic(::LinesearchStatus, ::Linesearch, params) = nothing

"""
    linesearch_warnings(status, ls, params=NullParameters())

Report a [`LinesearchStatus`](@ref) obtained from [`solve_with_status`](@ref). Compare this
to [`nonlinear_solver_warnings`](@ref). This is the *only* place where a line search emits
log messages, so every [`LinesearchMethod`](@ref) reports identically.

`LINESEARCH_FLOOR` and `LINESEARCH_STATIONARY` are reported only at `verbosity ≥ 2`, because both
are the *expected* final state of a converged solve — a residual that cannot be improved because it
is already as small as the arithmetic allows.

!!! info "Who this is for"
    This function is reached from [`solve`](@ref) and from nowhere else, which is what makes it
    safe for it to report unconditionally. A line search has two callers and owes them different
    things:

    - a **program** — a [`NonlinearSolver`](@ref), an optimizer — calls
      [`solve_with_status`](@ref), gets a [`LinesearchStatus`](@ref) it can act on, and gets no
      messages. It is not the user, and a diagnosis it can read is worth more to it than one it
      would have to scrape out of a log. What it does with the status is its own business:
      [`solver_step!`](@ref) tallies it (see [`record_linesearch!`](@ref)) and lets
      [`nonlinear_solver_warnings`](@ref) explain the solve once, at the end.
    - a **user** calls `solve`, which is this path, and a single call yields a single message.

    So there is nothing here to rate limit, and none of these messages carries a `maxlog`. They
    used to, because [`solver_step!`](@ref) called this function once per iteration and a solve
    that cannot make progress asks the line search for an impossible decrease at *every* one of
    them — thousands of identical messages. But `maxlog` is keyed on the source location of the
    `@warn`, so the caps were process-global and were never reset between `solve!` calls: once
    spent, they were spent for the rest of the session, and a genuine line-search failure in a
    later solve of a long run was silent. Not reporting from inside the loop removes the flood at
    its source, so the caps are gone and nothing goes permanently silent.

Whether an irreducible merit actually *matters* is the outer iteration's call, and
[`nonlinear_solver_warnings`](@ref) makes it: it reports stagnation once, naming the residual
that was achieved, the tolerance that was requested, and what the line search reported along the
way.

The messages themselves live in [`report_linesearch_status`](@ref) rather than here, which is a
compile-time rather than a stylistic decision — see its docstring before merging them back.
"""
function linesearch_warnings(status::LinesearchStatus, ls::Linesearch, params=NullParameters())
    # The two silent outcomes are filtered before the call, so that the path a healthy solve takes
    # on every iteration does not even copy the 27-field `Options` for the callee.
    oc = outcome(status)
    oc === LINESEARCH_DECREASED || oc === LINESEARCH_UNKNOWN ||
        report_linesearch_status(status, nameof(typeof(method(ls))), config(ls))

    verbosity(config(ls)) ≥ 2 && curvature_diagnostic(status, ls, params)

    nothing
end

# The two wordings of the `LINESEARCH_EXHAUSTED` message. With `αmin = 0` — every method other than
# `Backtracking` — the budget wording is selected, which is correct: those searches only ever
# exhaust by running out of budget or by failing to bracket, never by reaching an `αmin` floor.
# Called from *inside* the `@warn` message so that the string is built only for a message that is
# actually shown; see the `FLOOR` branch of the barrier below.
function linesearch_exhausted_reason(status::LinesearchStatus, config::Options)
    steplength(status) > status.αmin ?
    "the budget linesearch_max_iterations = $(config.linesearch_max_iterations) was spent, or the merit could not be bracketed" :
    "the merit changed by $(status.φ - status.φ₀) at the smallest informative step αmin = $(status.αmin), which exceeds the round-off resolution τ = $(status.τ), so φ'(0) = $(status.d₀) is inconsistent with the merit (a stale or regularized Jacobian, an inexact linear solve, or a non-smooth problem)"
end

"""
    report_linesearch_status(status, name, config)

Emit the messages for a [`LinesearchStatus`](@ref); the reporting half of
[`linesearch_warnings`](@ref), whose docstring documents the verbosity policy and who it is for.

# Implementation

This is a function barrier, and its signature is what makes it one. [`linesearch_warnings`](@ref)
is called from [`solve`](@ref) — every direct call to a line search, and nothing else since
[`solver_step!`](@ref) stopped reporting per iteration — and takes a [`Linesearch`](@ref), which
carries the closure types of its [`LinesearchProblem`](@ref), and a `NamedTuple` of parameters. So
it is specialized once per *problem* a line search is built for. A message in its body is
specialized with it, and all of the `Base.CoreLogging` and string-interpolation code that `@warn`
expands to is re-inferred and re-codegen'd for each one, which on a caller that builds one line
search per tableau dominates the cost of the calls themselves.

Taking `name` and `config`, and nothing whose type can vary per solver, bounds the specializations
of this function to one per element-type combination for the whole session.
[`nonlinear_solver_warnings`](@ref) and [`print_status`](@ref) have the same shape for the same
reason.

So: do not give this function a parameter whose type varies per solver, and do not move the
messages back into [`linesearch_warnings`](@ref). `test/linesearch_tests.jl` asserts both — the
first from the method signature, which *bounds* the specialization set rather than sampling it,
and the second by scanning the lowered code of each function for `Base.CoreLogging`.

The `@noinline` is a guard rather than the mechanism: Julia's inliner refuses a body this size
anyway, but a future one that is more willing would undo the barrier, and nothing in the caller
wants this inlined.

The element types are deliberately *not* tied together as `LinesearchStatus{T}`/`Options{T}`: this
is a reporting path, and a precision mismatch anywhere upstream should not turn a diagnostic into a
`MethodError` that replaces the problem being diagnosed. [`nonlinear_solver_warnings`](@ref) is
written the same way.
"""
@noinline function report_linesearch_status(status::LinesearchStatus, name::Symbol, config::Options)
    # `LINESEARCH_DECREASED` and `LINESEARCH_UNKNOWN` match none of the branches below, which is how
    # they stay silent; the chain deliberately has no `else`.
    oc = outcome(status)
    verbose = verbosity(config)

    if oc === LINESEARCH_FLOOR
        # Gated at `verbosity ≥ 2`, not 1: reaching the merit's round-off floor is the *normal*
        # final state of a converged solve (the residual cannot be improved because it is
        # already as small as the arithmetic allows), so warning about it at the default
        # verbosity means warning about success. Whether the floor matters is the outer
        # iteration's call, and it makes it: `record_stall!` counts a floor only while the
        # residual is *not* small, and `nonlinear_solver_warnings` then reports it once, with
        # the achieved residual and the requested tolerance.
        # `αmin` is a `Backtracking` quantity (zero means "not applicable", see `LinesearchStatus`),
        # so the clause naming it is only included when there is a value to name. It sits inside the
        # message rather than in a temporary before it: Julia evaluates a `@warn` message only once
        # the verbosity gate has passed, and the overwhelmingly common case is a caller running at
        # a verbosity that discards it, so a temporary would be built and thrown away every time.
        verbose ≥ 2 && @warn "$(name) line search: no trial step changed the merit by more than the round-off resolution τ = $(status.τ) in $(trials(status)) trial step(s). φ(0) = $(status.φ₀) has reached its round-off floor, so no step can decrease it$(iszero(status.αmin) ? "" : " (the smallest informative step is αmin = $(status.αmin))"). Returning α = $(steplength(status)). Check whether the requested residual tolerance is attainable in this precision."
    elseif oc === LINESEARCH_EXHAUSTED
        verbose ≥ 1 && @warn "$(name) line search: no step satisfied the sufficient decrease condition in $(trials(status)) trial step(s) — $(linesearch_exhausted_reason(status, config)). Returning α = $(steplength(status))."
    elseif oc === LINESEARCH_NO_DESCENT
        verbose ≥ 1 && @warn "$(name) line search: φ'(0) = $(status.d₀) (with φ(0) = $(status.φ₀)) is not a descent direction, so no α can satisfy the sufficient decrease condition. Returning α = $(steplength(status))."
    elseif oc === LINESEARCH_STATIONARY
        verbose ≥ 2 && @warn "$(name) line search: φ'(0) = 0, the merit is stationary at α = 0. Returning α = $(steplength(status))."
    end

    nothing
end
