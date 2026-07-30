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
- `LINESEARCH_UNKNOWN`: the method does not report an outcome (the generic fallback of
  [`solve_with_status`](@ref)).
"""
@enum LinesearchOutcome::Int8 begin
    LINESEARCH_DECREASED
    LINESEARCH_FLOOR
    LINESEARCH_EXHAUSTED
    LINESEARCH_NO_DESCENT
    LINESEARCH_STATIONARY
    LINESEARCH_UNKNOWN
end

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
- `trials`: the number of trial steps ``\alpha > 0`` at which the merit was actually
  evaluated — *not* the `linesearch_max_iterations` budget,
- `φ₀`, `d₀`: the merit and its derivative at the anchor ``\alpha = 0``,
- `φ`: the merit at the returned step,
- `τ`: the round-off allowance used in the [`SufficientDecreaseCondition`](@ref),
- `αmin`: the smallest step length that could still be decided by the merit rather than by
  rounding (see [`backtracking_αmin`](@ref)).
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
[`LinesearchOutcome`](@ref); the remaining diagnostics are filled with `NaN`/zero. Used by
the generic fallback of [`solve_with_status`](@ref) for methods that do not report them.
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
    check_anchor(φ₀, d₀, α)

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
Jacobian (see [`maybe_refactorize!`](@ref)), which is why the line search reports the situation
instead of trying to salvage a step from it.
"""
function check_anchor(φ₀::T, d₀::T, α::T) where {T}
    # A non-positive trial step is not a step the caller can be handed back, so substitute the
    # unit step — the same convention `StrongWolfe` uses for its initial trial. This is what
    # makes the α > 0 guarantee hold even when the caller passes α ≤ 0.
    αout = α > zero(T) ? α : one(T)
    if !isfinite(φ₀) || !isfinite(d₀) || d₀ > zero(T)
        LinesearchStatus{T}(αout, LINESEARCH_NO_DESCENT, 0, φ₀, d₀, φ₀, zero(T), zero(T))
    elseif iszero(d₀)
        LinesearchStatus{T}(αout, LINESEARCH_STATIONARY, 0, φ₀, d₀, φ₀, zero(T), zero(T))
    end
end

"""
    solve_with_status(ls, α, params=NullParameters())

Like [`solve`](@ref), but return a [`LinesearchStatus`](@ref) — the step length *plus* the
reason the search stopped — and emit no log messages. Use [`linesearch_warnings`](@ref) to
report the status.

Only [`Backtracking`](@ref) reports a genuine [`LinesearchOutcome`](@ref); the generic
fallback calls [`solve`](@ref) and reports `LINESEARCH_UNKNOWN`, so a caller may use
`solve_with_status` uniformly for every [`LinesearchMethod`](@ref).
"""
solve_with_status(ls::Linesearch{T}, α::T, params=NullParameters()) where {T} =
    LinesearchStatus(solve(ls, α, params))

"""
    curvature_diagnostic(status, ls, params)

Method-specific extra diagnostic emitted by [`linesearch_warnings`](@ref) at
`verbosity ≥ 2`. The fallback does nothing; [`Backtracking`](@ref) checks the
[`CurvatureCondition`](@ref), which costs a derivative evaluation — a full
[`Jacobian`](@ref) for the line search problem of a [`NonlinearSolver`](@ref), hence the
verbosity gate.
"""
curvature_diagnostic(::LinesearchStatus, ::Linesearch, params) = nothing

"""
    linesearch_warnings(status, ls, params=NullParameters())

Report a [`LinesearchStatus`](@ref) obtained from [`solve_with_status`](@ref). Compare this
to [`nonlinear_solver_warnings`](@ref). This is the *only* place where a line search emits
log messages, so [`solve`](@ref) and [`solver_step!`](@ref) report identically.

Two things keep this quiet in normal use. `LINESEARCH_FLOOR` and `LINESEARCH_STATIONARY` are
reported only at `verbosity ≥ 2`, because both are the *expected* final state of a converged
solve — a residual that cannot be improved because it is already as small as the arithmetic
allows. And the remaining outcomes are rate limited with `maxlog`, because a solve that cannot
make progress asks the line search for an impossible decrease at every one of its iterations,
which an unconditional warning turns into thousands of identical messages.

Whether an irreducible merit actually *matters* is the outer iteration's call, and
[`nonlinear_solver_warnings`](@ref) makes it: it reports stagnation once, naming the residual
that was achieved and the tolerance that was requested.
"""
function linesearch_warnings(status::LinesearchStatus, ls::Linesearch, params=NullParameters())
    verbose = config(ls).verbosity
    name = nameof(typeof(method(ls)))

    if outcome(status) === LINESEARCH_FLOOR
        # Gated at `verbosity ≥ 2`, not 1: reaching the merit's round-off floor is the *normal*
        # final state of a converged solve (the residual cannot be improved because it is
        # already as small as the arithmetic allows), so warning about it at the default
        # verbosity means warning about success. Whether the floor matters is the outer
        # iteration's call, and it makes it: `record_stall!` counts a floor only while the
        # residual is *not* small, and `nonlinear_solver_warnings` then reports it once, with
        # the achieved residual and the requested tolerance.
        verbose ≥ 2 && @warn "$(name) line search: no trial step changed the merit by more than the round-off allowance τ = $(status.τ) in $(trials(status)) trial step(s). φ(0) = $(status.φ₀) has reached its round-off floor, so no step can decrease it (the smallest informative step is αmin = $(status.αmin)). Returning α = $(steplength(status)). Check whether the requested residual tolerance is attainable in this precision." maxlog = 1
    elseif outcome(status) === LINESEARCH_EXHAUSTED
        reason = steplength(status) > status.αmin ?
                 "the budget linesearch_max_iterations = $(config(ls).linesearch_max_iterations) was spent" :
                 "the merit changed by $(status.φ - status.φ₀) at the smallest informative step αmin = $(status.αmin), which exceeds the round-off allowance τ = $(status.τ), so φ'(0) = $(status.d₀) is inconsistent with the merit (a stale or regularized Jacobian, an inexact linear solve, or a non-smooth problem)"
        verbose ≥ 1 && @warn "$(name) line search: no step satisfied the sufficient decrease condition in $(trials(status)) trial step(s) — $(reason). Returning α = $(steplength(status))." maxlog = 3
    elseif outcome(status) === LINESEARCH_NO_DESCENT
        verbose ≥ 1 && @warn "$(name) line search: φ'(0) = $(status.d₀) (with φ(0) = $(status.φ₀)) is not a descent direction, so no α can satisfy the sufficient decrease condition. Returning α = $(steplength(status))." maxlog = 3
    elseif outcome(status) === LINESEARCH_STATIONARY
        verbose ≥ 2 && @warn "$(name) line search: φ'(0) = 0, the merit is stationary at α = 0. Returning α = $(steplength(status))."
    end

    verbose ≥ 2 && curvature_diagnostic(status, ls, params)

    nothing
end
