@doc raw"""
    NonlinearSolverStatus

Stores absolute and successive residuals for `x` and `f`. It is used as a diagnostic tool in [`NewtonSolver`](@ref).

!!! info
    Compare this to the [`NonlinearSolverState`](@ref) and the [`NonlinearSolverCache`](@ref).

# Keys
- `iterations`: number of iterations
- `stalls`: number of *consecutive* stalled steps, see [`stalled_step`](@ref) and [`isstalled`](@ref),
- `iterations_since_progress`: iterations since the residual last dropped by `f_stall_factor`, see [`iterations_since_progress`](@ref),
- `rxₛ`: successive residual in `x`,
- `rfₐ`: absolute residual in `f`,
- `rfₛ`: successive residual in `f`,
- `x_converged::Bool`
- `f_converged::Bool`
- `f_increased::Bool`
- `stalled::Bool`: the *last* step stalled, see [`stalled_step`](@ref)
- `not_progressing::Bool`: the iteration is not getting anywhere, see [`no_progress`](@ref)
- `ls_outcomes`: how often the line search reported each [`LinesearchOutcome`](@ref) during the
  solve, indexed by [`linesearch_index`](@ref); see [`linesearch_outcomes`](@ref) and
  [`linesearch_failures`](@ref)

!!! info "The line-search tally is the programmatic channel"
    A line search does not log from inside a solve — it reports to the solver, which accumulates
    the outcomes here (see [`record_linesearch!`](@ref)). A caller that wants to *act* on a
    rejected line search rather than read about it — restart an approximate Hessian, fall back to
    steepest descent — reads this tally instead of scraping the log. That is what
    [`solve_with_status!`](@ref) is for.

# Examples

```jldoctest; setup = :(using SimpleSolvers: NonlinearSolverStatus, NonlinearSolverState, NonlinearSolverCache, Options)
x = [1., 2., 3., 4.]
state = NonlinearSolverState(x)
cache = NonlinearSolverCache(x, x)
config = Options()
NonlinearSolverStatus(state, config)

# output

i=   0,
rxₛ= NaN,
rfₐ= NaN,
rfₛ= NaN
```
"""
struct NonlinearSolverStatus{T}
    iterations::Int
    stalls::Int
    iterations_since_progress::Int

    rxₛ::T
    rfₐ::T
    rfₛ::T

    x_converged::Bool
    f_converged::Bool
    f_increased::Bool
    stalled::Bool
    not_progressing::Bool

    # An `NTuple` and not the `MVector` the state keeps: the status is immutable and is built more
    # than once per iteration (see `record_progress!`), so it takes a snapshot rather than a handle
    # on a buffer that keeps changing under it.
    ls_outcomes::NTuple{NLINESEARCH_OUTCOMES, Int}
end

"""
    linesearch_outcomes(status)

Return the tally of [`LinesearchOutcome`](@ref)s the line search reported during the solve,
indexed by [`linesearch_index`](@ref). See [`record_linesearch!`](@ref) for why this, and not the
log, is how a caller learns that the line search was rejected.

# Examples

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: linesearch_outcomes, linesearch_index)
julia> F(y, x, params) = y .= x .^ 2 .- 2;

julia> x = [1.0];

julia> st = solve_with_status!(x, NonlinearProblem(F, zero(x)), Newton(); verbosity = 0);

julia> linesearch_outcomes(st)[linesearch_index(LINESEARCH_NO_DESCENT)]
0
```
"""
linesearch_outcomes(status::NonlinearSolverStatus) = status.ls_outcomes

"""
    linesearch_failures(status)

The number of steps whose line search reported a failure, i.e. [`linesearch_outcomes`](@ref)
summed over the outcomes that are not [`isbenign`](@ref).
"""
function linesearch_failures(status::NonlinearSolverStatus)
    sum(oc -> isbenign(oc) ? 0 : status.ls_outcomes[linesearch_index(oc)], instances(LinesearchOutcome))
end

"""
    dominant_linesearch_outcome(status, count_floor=true)

The non-[`isbenign`](@ref) [`LinesearchOutcome`](@ref) the line search reported most often during
the solve, or `nothing` if it reported none. This is what
[`nonlinear_solver_warnings`](@ref) names when it explains a solve that did not converge: a solve
whose line search failed does so for one reason nearly every time, and naming that one reason is
more use than a histogram.

Ties go to the outcome declared first in [`LinesearchOutcome`](@ref), which orders them from the
benign end (`LINESEARCH_FLOOR`, the merit is simply irreducible) towards the actionable one
(`LINESEARCH_NO_DESCENT`, the direction is wrong and the [`Jacobian`](@ref) is the suspect) — so a
tie is broken *away* from the more alarming diagnosis rather than towards it.

`count_floor = false` skips `LINESEARCH_FLOOR`, which is how a *converged* solve is asked whether
anything went wrong: reaching the merit's round-off floor on the last step is how a solve converges,
so for that question the floor is not a failure at all. It is the tie rule above that makes the
distinction matter — a converged solve that floored once and was exhausted once would otherwise be
explained by the floor, which is the half of it that is expected.
"""
function dominant_linesearch_outcome(status::NonlinearSolverStatus, count_floor::Bool = true)
    best = nothing
    count = 0
    for oc in instances(LinesearchOutcome)
        (isbenign(oc) || (!count_floor && oc === LINESEARCH_FLOOR)) && continue
        n = status.ls_outcomes[linesearch_index(oc)]
        n > count && ((best, count) = (oc, n))
    end
    best
end

@doc raw"""
    residuals(state)

Compute the residuals for `state::`[`NonlinearSolverState`](@ref).
The computed residuals are the following:
- `rxₛ` : successive residual (the norm of ``x - \bar{x}``),
- `rfₐ`: absolute residual in ``f``,
- `rfₛ` : successive residual (the norm of ``y - \bar{y}``).
"""
function residuals(state::NonlinearSolverState)
    rxₛ = l2norm(solution(state), previoussolution(state))
    rfₐ = l2norm(value(state))
    rfₛ = l2norm(value(state), previousvalue(state))

    rxₛ, rfₐ, rfₛ
end

"""
    assess_convergence(rxₛ, rfₐ, rfₛ, config, state)

Assess convergence for `status::`[`NonlinearSolverStatus`](@ref) and return the
triple `(x_converged, f_converged, f_increased)`.

The successive-change criteria (in `x` and `f`) alone are *not* sufficient to
declare convergence: a stalled step (e.g. an artificially tiny line-search step)
makes the successive residuals `rxₛ` and `rfₛ` vanish even when the absolute
residual `rfₐ` is large.  We therefore require the residual to be *small* — written
`residual_small` below — *in addition* to the successive-change criterion before
reporting convergence.  The residual passes the standard `atol + rtol·‖F₀‖` test:

`residual_small` ⟺ `rfₐ ≤ config.f_abstol + config.f_reltol * initial_residual(state)`,

with the absolute tolerance `atol = config.f_abstol` (defaulting to `0`) and the relative
tolerance `rtol = config.f_reltol` (defaulting to `√eps(T)`) applied to the initial residual
`‖F(x₀)‖`. Concretely:

- `x_converged`: `rxₛ ≤ norm(solution(state)) * config.x_suctol` **and** `residual_small`,
- `f_converged`: (`rfₛ ≤ norm(value(state)) * config.f_suctol` **and** `residual_small`) **or** `rfₐ ≤ config.f_abstol`,
- `f_increased`: `norm(value(state)) > norm(previousvalue(state))`.

This guards the successive-change criteria against stagnation: it is loose enough that a
genuinely converged iterate satisfies it (the successive-change criteria still supply the
tight, machine-precision accuracy) yet tight enough to reject a step that stalls near its
initial residual (`rfₐ ≈ ‖F(x₀)‖ ≫ f_reltol·‖F(x₀)‖`). The relative term is what lets a
*well-scaled* solve whose residual floors at a large *absolute* value (e.g. a
large-magnitude or ill-conditioned `F`) still converge; it drops to zero (leaving the pure
absolute `f_abstol` test) until the state has been initialized (`initial_residual` is `NaN`).

Also see [`meets_stopping_criteria`](@ref).
"""
function assess_convergence(
        rxₛ::Number, rfₐ::Number, rfₛ::Number, config::Options, state::NonlinearSolverState)
    # The iterate/value has stopped changing (successive-change criteria).
    x_settled = iterate_settled(rxₛ, config, state)
    f_settled = rfₛ ≤ norm(value(state)) * config.f_suctol

    small = residual_small(rfₐ, config, state)

    x_converged = x_settled && small
    f_converged = (f_settled && small) || rfₐ ≤ config.f_abstol

    f_increased = norm(value(state)) > norm(previousvalue(state))

    # The iterate froze without the residual becoming small: no progress is possible along
    # the current direction. See `stalled_step`.
    stalled = x_settled && !small

    x_converged, f_converged, f_increased, stalled
end

@doc raw"""
    residual_small(rfₐ, config, state)

Return `true` when the absolute residual `rfₐ` passes the standard
``\mathrm{atol} + \mathrm{rtol}\cdot\|F(x_0)\|`` residual test,

```math
r^f_a \leq \texttt{f\_abstol} + \texttt{f\_reltol}\cdot\|F(x_0)\|,
```

with ``\|F(x_0)\|`` the [`initial_residual`](@ref) of `state`. This lets a large-magnitude or
ill-conditioned solve converge once its residual is reduced by `f_reltol` from
``\|F(x_0)\|``, while a step that stalls near ``\|F(x_0)\|`` still fails. The relative term
drops to zero until the state has been initialized (`initial_residual` is `NaN`), leaving the
pure absolute `f_abstol` test.

This gate is shared by [`assess_convergence`](@ref), which requires it *in addition* to a
successive-change criterion, and by [`stalled_step`](@ref), which requires its *negation*: a
frozen iterate is convergence when the residual is small and stagnation when it is not. The
two are therefore mutually exclusive by construction.
"""
function residual_small(rfₐ::Number, config::Options, state::NonlinearSolverState)
    r₀ = initial_residual(state)
    # `isfinite`, not `!isnan`: an *infinite* initial residual would make the relative term `Inf`
    # and this gate vacuously true, so every finite residual — a residual of 1e10 included —
    # would count as small and the solve would report convergence wherever it happened to land.
    # An overflowed `‖F(x₀)‖` is no more a usable reference scale than an uninitialized one.
    relative_residual = isfinite(r₀) ? config.f_reltol * r₀ : zero(rfₐ)
    rfₐ ≤ config.f_abstol + relative_residual
end

"""
    iterate_settled(rxₛ, config, state)

Return `true` when the last step did not move the iterate, `rxₛ ≤ ‖x‖·x_suctol`. Used by
[`assess_convergence`](@ref) and [`stalled_step`](@ref).

An infinite step never counts as settled, even though `Inf ≤ ‖x‖·x_suctol` holds once `‖x‖` has
overflowed too: an iterate that jumped to infinity has neither converged nor frozen, it has
broken down, and that is what [`havenonfinite`](@ref) is for.
"""
function iterate_settled(rxₛ::Number, config::Options, state::NonlinearSolverState)
    isfinite(rxₛ) && rxₛ ≤ norm(solution(state)) * config.x_suctol
end

@doc raw"""
    stalled_step(rxₛ, rfₐ, config, state)

Return `true` when the last step *stalled*: it left the iterate unchanged (see
[`iterate_settled`](@ref)) while the residual is **not** small (see
[`residual_small`](@ref)).

A stalled step is the failure mode that the residual gate in [`assess_convergence`](@ref)
correctly refuses to call convergence; this predicate is what makes it visible. The step
length ``\alpha\|d\|`` has dropped below the round-off level of ``x``, so the merit
``\|F\|^2`` cannot be reduced along the current direction — typically because the requested
`f_abstol` lies *below the round-off floor of the residual itself*. Taking another step
recomputes the same direction and the same negligible ``\alpha``, so the iteration would spin
all the way to `max_iterations`, asking the line search on every one of those steps to improve
a residual that is already pure round-off noise.

[`meets_stopping_criteria`](@ref) therefore stops after `config.max_stalls` *consecutive*
stalled steps (counted by [`record_stall!`](@ref)), and
[`nonlinear_solver_warnings`](@ref) reports the achieved residual against the requested
tolerance instead of the misleading `"Solver took 1000 iterations."`.

!!! info
    The condition is deliberately phrased in terms of the step actually taken rather than a
    line-search return code. It is the same diagnosis for a [`Backtracking`](@ref) ladder that
    exhausted, a [`StrongWolfe`](@ref) search that found no acceptable step, a
    [`Static`](@ref) step along an underflowed direction, a [`DogLegSolver`](@ref) whose
    trust-region radius collapsed and a [`PicardSolver`](@ref) whose fixed-point map is
    locally expanding. A line search that *knows* it is at the round-off floor can report one
    iteration earlier via [`flag_stall!`](@ref).
"""
function stalled_step(rxₛ::Number, rfₐ::Number, config::Options, state::NonlinearSolverState)
    iterate_settled(rxₛ, config, state) && !residual_small(rfₐ, config, state)
end

@doc raw"""
    no_progress(rfₐ, config, state)

Return `true` when the iteration has spent `config.f_stall_window` iterations without the
residual dropping by `config.f_stall_factor` (see [`iterations_since_progress`](@ref)) while the
residual is **not** small (see [`residual_small`](@ref)). Always `false` at the default
`f_stall_window = 0`, which disables the criterion — see [`F_STALL_WINDOW`](@ref) for why it is
opt-in.

This is the sibling of [`stalled_step`](@ref) for a solve whose iterate has *not* frozen. Both
end an iteration that cannot reach the requested tolerance, and they cover disjoint cases:
`stalled_step` fires when the step has dropped below the round-off level of ``x``, so that the
merit cannot be reduced *along the current direction*; `no_progress` fires when the steps are
perfectly healthy and the residual is descending — just towards a floor above the tolerance,
slowly enough that the remaining budget cannot get there. Their thresholds differ by orders of
magnitude for the same reason: two consecutive stalled steps are conclusive because the second
one had a fresh [`Jacobian`](@ref), whereas no number of *moving* steps is conclusive about a
rate, which is why one is a default and the other a policy the caller sets.

The `!residual_small` gate is the same one [`stalled_step`](@ref) carries, and it is what keeps
giving up and converging mutually exclusive: a residual that has stopped improving *because it
is already small enough* is success, and [`assess_convergence`](@ref) says so.
"""
function no_progress(rfₐ::Number, config::Options, state::NonlinearSolverState)
    config.f_stall_window > 0 &&
        iterations_since_progress(state) ≥ config.f_stall_window &&
        !residual_small(rfₐ, config, state)
end

"""
    record_stall!(state, config)
    record_stall!(state, config, rxₛ, rfₐ)

Update the consecutive-stall counter of `state::`[`NonlinearSolverState`](@ref): increment it
when the last step [`stalled_step`](@ref) *or* the line search flagged a stall (see
[`flag_stall!`](@ref)), and reset it to zero otherwise. The flag is cleared either way.
Returns the new count (see [`stall_number`](@ref)). The four-argument form takes residuals the
caller has already computed; the two-argument form computes them from the state.

This is a per-iteration measurement rather than a predicate, so it must be called exactly once
per iteration; [`record_iteration!`](@ref) is what does so, and carries that contract. That is
why the counter is not maintained inside [`assess_convergence`](@ref) or
[`NonlinearSolverStatus`](@ref): those are pure and are evaluated more than once per iteration,
so incrementing there would double-count. A hand-rolled iteration that drives
[`solver_step!`](@ref) directly and never records simply keeps the count at zero and behaves
exactly as before.
"""
function record_stall!(state::NonlinearSolverState, config::Options)
    rxₛ, rfₐ, _ = residuals(state)
    record_stall!(state, config, rxₛ, rfₐ)
end

function record_stall!(state::NonlinearSolverState, config::Options, rxₛ::Number, rfₐ::Number)
    flagged = state.stallflag
    state.stallflag = false
    # The line-search flag substitutes for `iterate_settled` (it is the same news, one
    # iteration earlier), but it is gated by the *same* residual test: at convergence the
    # merit is also at its round-off floor, and that is success, not stagnation. Keeping the
    # gate is what makes stagnation and convergence mutually exclusive (see `isstalled`).
    stalled = (flagged || iterate_settled(rxₛ, config, state)) &&
              !residual_small(rfₐ, config, state)
    state.stalls = stalled ? state.stalls + 1 : 0
end

"""
    record_iteration!(state, config)

Take the two per-iteration measurements of `state::`[`NonlinearSolverState`](@ref) — the
consecutive-stall counter ([`record_stall!`](@ref)) and the progress reference
([`record_progress!`](@ref)) — from a single evaluation of [`residuals`](@ref).

This is the one function carrying the "exactly once per iteration" contract that both counters
depend on: [`solve!`](@ref) calls it right after [`update!`](@ref), and nothing else does. Both
counters are increments rather than predicates, so calling it twice would double-count and
never calling it leaves both at zero, which is exactly how a hand-rolled iteration that drives
[`solver_step!`](@ref) directly behaves.

Sharing the residuals is why it exists at all: the two recordings need `rxₛ` and `rfₐ` between
them, and computing them once here rather than once in each keeps a per-iteration norm off the
hot loop.
"""
function record_iteration!(state::NonlinearSolverState, config::Options)
    rxₛ, rfₐ, _ = residuals(state)
    record_stall!(state, config, rxₛ, rfₐ)
    record_progress!(state, config, rfₐ)
end

function NonlinearSolverStatus(state::NonlinearSolverState{T}, config::Options{T}) where {T}
    rxₛ, rfₐ, rfₛ = residuals(state)
    x_converged, f_converged, f_increased, stalled = assess_convergence(
        rxₛ, rfₐ, rfₛ, config, state)
    # `no_progress` is evaluated here rather than in `assess_convergence` because it is not a
    # convergence question: it reads a measurement taken once per iteration by `record_progress!`
    # instead of the residuals of the current step.
    NonlinearSolverStatus{T}(
        iteration_number(state), stall_number(state), iterations_since_progress(state),
        rxₛ, rfₐ, rfₛ, x_converged, f_converged, f_increased,
        stalled, no_progress(rfₐ, config, state),
        Tuple(linesearch_outcomes(state)))
end

# The stall and no-progress lines are appended only when they are relevant, so the printout of a
# fresh status is unchanged. `spent_without_progress` rather than its proportion alone is what
# keeps a *healthy* solve's printout unchanged too: this is the one caller that cannot check the
# budget, having no `Options` — see `F_STALL_REPORT_MINIMUM`.
function Base.show(io::IO, status::NonlinearSolverStatus)
    print(io,
        (@sprintf "i=%4i" status.iterations), ",\n",
        (@sprintf "rxₛ=%4e" status.rxₛ), ",\n",
        (@sprintf "rfₐ=%4e" status.rfₐ), ",\n",
        (@sprintf "rfₛ=%4e" status.rfₛ),
        status.stalls > 0 ? ",\n" * (@sprintf "stalls=%4i" status.stalls) : "",
        spent_without_progress(status) ?
        ",\n" * (@sprintf "no progress for=%4i" status.iterations_since_progress) : "")
end

@doc raw"""
    print_status(status, config)

Print the solver status if:
- `config.verbosity` ``\geq1`` and one of the following three
1. the solver is converged,
2. `status.iterations ≥ config.max_iterations`,
3. `status.iterations ≥ config.warn_iterations`
- `config.verbosity` ``>1.``
"""
function print_status(status::NonlinearSolverStatus, config::Options)
    if (config.verbosity ≥ 1 &&
        (isconverged(status) || status.iterations ≥ config.max_iterations ||
         status.iterations ≥ config.warn_iterations)) ||
       config.verbosity > 1
        println(status)
    end
end

"""
    isconverged(status)

Check if either `x` or `f` has converged.

The `status` is a [`NonlinearSolverStatus`](@ref).
"""
isconverged(status::NonlinearSolverStatus) = status.x_converged || status.f_converged

"""
    havenonfinite(status)

Check whether any of the three residuals of a [`NonlinearSolverStatus`](@ref) — `rxₛ`, `rfₐ`,
`rfₛ` — is not finite, i.e. whether the iteration has left the region where the problem is
representable. Used by [`meets_stopping_criteria`](@ref) to give up and by
[`nonlinear_solver_warnings`](@ref) to say so.

The test is `isfinite`, not `!isnan`: a residual that has *overflowed* is as unusable as an
undefined one, and a pure-`NaN` test misses it entirely. Nothing else catches it either — the
`rfₐ > f_abstol_break` gate does not fire for an infinite residual, since `f_abstol_break`
defaults to `Inf` and `Inf > Inf` is false — so such a solve would run its whole `max_iterations`
budget with no diagnosis at all. The solver-side guards are widened the same way; see
[`nan_recovery!`](@ref).

Note that a status is never *converged* by accident here — every comparison with `NaN` is false
and no infinite residual passes [`residual_small`](@ref) — so this predicate decides when to
stop and what to report, not whether the answer is good.
"""
function havenonfinite(status::NonlinearSolverStatus)
    !(isfinite(status.rxₛ) && isfinite(status.rfₐ) && isfinite(status.rfₛ))
end

"""
    isstalled(status, config)

Check whether the iteration has *stagnated*: `config.max_stalls` consecutive steps
[`stalled_step`](@ref).

Mutually exclusive with [`isconverged`](@ref) — a stalled step is by definition one whose
residual is *not* small, whereas both convergence branches require that it is.

A stagnated solve has reached the numerical floor of its residual and cannot improve it.
Whether that counts as success is the caller's decision, which is why the status is queryable
(see [`status`](@ref)): if `status.rfₐ` is acceptable to *you*, treat `isstalled` as success —
and consider raising `f_abstol` above it, since the tolerance you asked for is not attainable.
"""
function isstalled(status::NonlinearSolverStatus, config::Options)
    status.stalls ≥ config.max_stalls
end

"""
    isnotprogressing(status)

Check whether the iteration has been given up on for lack of *progress*: `config.f_stall_window`
iterations without the residual dropping by `config.f_stall_factor`, see [`no_progress`](@ref).
Always `false` at the default `f_stall_window = 0`.

Mutually exclusive with [`isconverged`](@ref), for the same reason [`isstalled`](@ref) is: the
criterion requires the residual *not* to be small, whereas both convergence branches require that
it is.

As with [`isstalled`](@ref), whether this counts as failure is the caller's decision: the solve
reached `status.rfₐ` and could not do better within the window it was given.
"""
isnotprogressing(status::NonlinearSolverStatus) = status.not_progressing

"""
    spent_without_progress(status)

Check whether the iteration failed to converge and spent at least *half* of its iterations, and
at least [`F_STALL_REPORT_MINIMUM`](@ref) of them, without the residual dropping by
`config.f_stall_factor` — the diagnosis [`nonlinear_solver_warnings`](@ref) reports when a solve
has used its whole budget, and the condition under which the no-progress line is shown by `show`.

This is not gated on any option, unlike [`isnotprogressing`](@ref), and it can afford not to be
because it is only ever used to *describe* a solve, never to decide one:
[`nonlinear_solver_warnings`](@ref) consults it about a solve that has already spent
`max_iterations`, and `show` about one it has been handed to print. A threshold that would be
reckless as a stopping criterion (see [`F_STALL_WINDOW`](@ref)) is harmless as an explanation —
which is why the two exist separately.

Both guards are here rather than at the call sites because `show` has no `Options` and so cannot
apply them itself, and it is the caller most exposed to a false positive. [`isconverged`](@ref)
is the primary one: a residual that stopped improving *because it was already small enough* is
success, and a solve held to a large `min_iterations` would otherwise spend most of its
iterations on a converged plateau and start explaining itself. The absolute minimum is the
backstop for a solve that has *not* converged and is simply short — without it the proportion
alone is satisfied by a *two*-iteration solve whose last step did not halve the residual. With
both, no healthy solve comes close: a `Gauss(2)` Lotka-Volterra run converges in two to four
iterations with at most one of them unproductive.

A long healthy solve does not reach it either, for the separate reason that its residual keeps
halving: an iteration converging linearly with rate ``\\rho`` halves every ``-1/\\log_2\\rho``
iterations, 69 of them even at ``\\rho = 0.99``.
"""
function spent_without_progress(status::NonlinearSolverStatus)
    !isconverged(status) &&
        status.iterations_since_progress ≥ F_STALL_REPORT_MINIMUM &&
        2 * status.iterations_since_progress ≥ status.iterations
end

"""
    meets_stopping_criteria(state, config)

Determines whether the iteration stops based on the current [`NonlinearSolverState`](@ref).

!!! warning
    The function `meets_stopping_criteria` may return `true` even if the solver has not converged. To check convergence, call [`assess_convergence`](@ref) (with the same input arguments).

The function `meets_stopping_criteria` returns `true` if one of the following is satisfied:
- the `status::`[`NonlinearSolverStatus`](@ref) is converged (checked with [`isconverged`](@ref)) and `state.iterations ≥ config.min_iterations`,
- the `status` has *stagnated* (checked with [`isstalled`](@ref), i.e. `config.max_stalls` consecutive steps that did not move the iterate while the residual is not small) and `state.iterations ≥ config.min_iterations`,
- the `status` is making no *progress* (checked with [`isnotprogressing`](@ref), i.e. `config.f_stall_window` iterations without the residual dropping by `config.f_stall_factor` while it is not small) and `state.iterations ≥ config.min_iterations`; this is opt-in and never fires at the default `f_stall_window = 0`,
- `status.f_increased` and `config.allow_f_increases = false` (i.e. `f` increased even though we do not allow it),
- `state.iterations ≥ config.max_iterations`,
- `status.rfₐ > config.f_abstol_break` (by default `Inf`). In theory this returns `true` if the residual gets too big.
- one of the residuals (`rxₛ`, `rfₐ`, `rfₛ`) is not finite (checked with [`havenonfinite`](@ref)) and `state.iterations ≥ 1`,
So convergence is only one possible criterion for which [`meets_stopping_criteria`](@ref). We may also satisfy a stopping criterion without having convergence!

# Examples

In the following example we show that `meets_stopping_criteria` evaluates to true when used on a freshly allocated [`NonlinearSolverStatus`](@ref):
```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: NonlinearSolverStatus, meets_stopping_criteria, NonlinearSolverCache, NonlinearSolverState)
julia> config = Options(verbosity=0);

julia> x = [NaN, 2., 3.]
3-element Vector{Float64}:
 NaN
   2.0
   3.0

julia> f = [NaN, 10., 20.]
3-element Vector{Float64}:
 NaN
  10.0
  20.0

julia> cache = NonlinearSolverCache(x, copy(x));

julia> state = NonlinearSolverState(x);

julia> update!(state, x, f); state.iterations += 1
1

julia> status = NonlinearSolverStatus(state, config);

julia> meets_stopping_criteria(state, config)
true
```
This obviously has not converged. To check convergence we can use [`assess_convergence`](@ref).
```
"""
function meets_stopping_criteria(state::NonlinearSolverState, config::Options)
    status = NonlinearSolverStatus(state, config)

    (isconverged(status) && state.iterations ≥ config.min_iterations) ||
        (isstalled(status, config) && state.iterations ≥ config.min_iterations) ||
        (isnotprogressing(status) && state.iterations ≥ config.min_iterations) ||
        (status.f_increased && !config.allow_f_increases) ||
        state.iterations ≥ config.max_iterations ||
        status.rfₐ > config.f_abstol_break ||
        (havenonfinite(status) && state.iterations ≥ 1)
end

# The two wordings of the no-progress message: the opt-in `f_stall_window` gave up, or the whole
# budget was spent and this explains what it was spent on. Called from *inside* the `@warn` message
# so that the string is built only for a message that is actually shown; see
# `linesearch_exhausted_reason`, which is factored out for the same reason.
function no_progress_reason(status::NonlinearSolverStatus, config::Options)
    isnotprogressing(status) ?
    "gave up after $(status.iterations) iterations: the residual rfₐ = $(status.rfₐ) did not improve by the factor f_stall_factor = $(config.f_stall_factor) in the last $(status.iterations_since_progress) of them, which is the f_stall_window = $(config.f_stall_window) you asked it to give up after" :
    "spent its full budget of max_iterations = $(config.max_iterations) iterations without converging: the residual rfₐ = $(status.rfₐ) did not improve by the factor f_stall_factor = $(config.f_stall_factor) in the last $(status.iterations_since_progress) of them, so either it is on a floor this problem imposes — in which case a larger budget will not help — or it is converging far too slowly for the budget it was given. Set f_stall_window to stop at that point instead of spending the whole budget"
end

@doc raw"""
    linesearch_reason(status, config, oc=dominant_linesearch_outcome(status))

The clause [`nonlinear_solver_warnings`](@ref) appends to explain a failed solve in terms of what
its line search reported, or `""` when it reported nothing but success. `oc` is the outcome to
explain — by default the dominant one, and for a converged solve the dominant one *other than*
`LINESEARCH_FLOOR`, which is what made that message fire (see
[`dominant_linesearch_outcome`](@ref)); the clause has to name the outcome the caller acted on,
not a different one that happens to be as frequent.

This is the *only* thing said about the line search during a solve. A line search does not log
from inside the iteration — it reports to the solver, which tallies the outcomes (see
[`record_linesearch!`](@ref)) — so without this clause a solve that stagnated because every one of
its steps was rejected would name the symptom and not the cause. A count is what makes it
evidence: "the line search rejected 194 of 200 steps" is a diagnosis, "the line search failed once"
is noise from the last step of an otherwise healthy solve.

Like `no_progress_reason` and `linesearch_exhausted_reason`, this is called from
*inside* the `@warn` message so the string is built only for a message that is actually shown.
"""
function linesearch_reason(status::NonlinearSolverStatus, config::Options,
        oc::Union{LinesearchOutcome, Nothing} = dominant_linesearch_outcome(status))
    isnothing(oc) && return ""
    n = linesearch_outcomes(status)[linesearch_index(oc)]
    " The line search reported $(oc) on $(n) of the $(status.iterations) step(s)" *
    (oc === LINESEARCH_NO_DESCENT ?
     ", i.e. φ'(0) > 0 — the direction was not a descent direction at all, which points at the Jacobian rather than at the tolerance: a stale one under refactorize > 1, a nonzero regularization_factor, or an inexact linear solve." :
     oc === LINESEARCH_EXHAUSTED ?
     ", i.e. the merit does vary but no trial step was acceptable, so either φ'(0) is inconsistent with φ (a stale or regularized Jacobian, an inexact linear solve, or a non-smooth problem) or linesearch_max_iterations = $(config.linesearch_max_iterations) was too small." :
     ", i.e. no trial step changed the merit by more than its round-off resolution — the same floor this message reports one level up.")
end

"""
    nonlinear_solver_warnings(status, config)

Report a [`NonlinearSolverStatus`](@ref) at the end of a [`solve!`](@ref): the iteration count
if it reached `warn_iterations`, *stagnation* at the residual floor (see [`isstalled`](@ref)
and [`stalled_step`](@ref)), a lack of *progress* (see [`spent_without_progress`](@ref) and
[`isnotprogressing`](@ref)), a disallowed residual increase, a residual beyond
`f_abstol_break`, and non-finite residuals (see [`havenonfinite`](@ref)). Compare this to
[`linesearch_warnings`](@ref), which does the same for the inner line search, and to
[`print_status`](@ref).

All messages except the iteration count and the two hard-failure ones are gated on
`config.verbosity ≥ 1`.

The three "this solve did not do what you asked" messages are mutually exclusive, most specific
first: stagnation (the iterate froze) wins over lack of progress (the iterate moves but the
residual is going nowhere), which in turn replaces the bare iteration count — which on its own
names a symptom and no cause.

# The line search reports here, not from inside the loop

A line search emits nothing during a solve: it reports to the solver through the
[`LinesearchStatus`](@ref) that [`solve_with_status`](@ref) returns, and [`solver_step!`](@ref)
tallies the outcomes (see [`record_linesearch!`](@ref)). So the two failure messages above carry
the clause [`linesearch_reason`](@ref) builds, which names the outcome the line search reported
most often and how often — the cause behind the symptom they otherwise report on their own. A
solve that converged anyway says the same thing at `verbosity ≥ 2`, as an `@info`, and only for a
failure that is *not* `LINESEARCH_FLOOR`: the last step of a converged solve reaches the merit's
round-off floor as a matter of course, so counting that would report every healthy solve.

To *act* on the outcome rather than read about it, use [`solve_with_status!`](@ref) and the tally
on the returned [`NonlinearSolverStatus`](@ref); see [`linesearch_outcomes`](@ref).

# Rate limiting

The three repeatable messages are gated on [`should_report!`](@ref), which reports the 1st, 2nd,
4th, 8th … occurrence of a diagnosis rather than the first three and then nothing ever again. The
keys of the two *diagnoses* carry the dominant line-search outcome, so a solve that starts failing
for a new reason is reported at once; the bare iteration count has no cause to key on and uses a
plain one. The trade-off — a *repeating* diagnosis is not reported on every occurrence — is spelled
out in that docstring. `verbosity = 0` still silences the solver completely.
"""
function nonlinear_solver_warnings(status::NonlinearSolverStatus, config::Options)
    # Stagnation is the more specific diagnosis and is reported below, so it suppresses the
    # no-progress one. Otherwise: either the caller opted into `f_stall_window` and it fired, or the
    # solve spent its whole budget and `spent_without_progress` says what on. (That predicate
    # carries its own `!isconverged` gate, so it is not repeated here.)
    stagnated = isstalled(status, config)
    noprogress = !stagnated &&
                 (isnotprogressing(status) ||
                  (status.iterations ≥ config.max_iterations &&
                   spent_without_progress(status)))

    # A solve that failed usually failed the same way at every step, so the dominant line-search
    # outcome is both the cause worth naming and the right thing to key the backoff on: a solve that
    # starts failing for a *new* reason is a new diagnosis and is reported at once, rather than
    # inheriting the suppressed counter of the old one.
    lskey = something(dominant_linesearch_outcome(status), :none)

    # The bare count names a symptom and no cause, so either of the two diagnoses below replaces it
    # rather than joining it — the mutual exclusivity the docstring promises.
    # Rate limited for the same reason the messages below are: a caller that drives `solve!` in a
    # loop would otherwise get this once per step for as long as the problem stays unattainable.
    (config.warn_iterations > 0 && status.iterations ≥ config.warn_iterations &&
     !noprogress && !stagnated && should_report!(:iterations)) &&
        (@warn "Solver took $(status.iterations) iterations.")
    # Same shape as the stagnation message: say what the solve achieved, what was asked of it, and
    # how to make the request attainable. The distinguishing news is that the iterate is *not*
    # stuck — the steps are healthy and the residual is simply not heading for the tolerance —
    # which points at the problem rather than at the solver or the precision.
    (noprogress && config.verbosity ≥ 1 && should_report!(Symbol(:noprogress_, lskey))) &&
        (@warn "Nonlinear solver $(no_progress_reason(status, config)). The requested residual tolerance was f_abstol = $(config.f_abstol) (plus f_reltol = $(config.f_reltol) times the initial residual ‖F(x₀)‖).$(status.stalled ? "" : " The iterate has not frozen (the last step was rxₛ = $(status.rxₛ)), so this is not the round-off floor of x that stagnation detection reports; a residual that is not heading for the tolerance while the steps are healthy usually means a floor of the problem itself — a model, discretisation or ansatz error that F cannot resolve, and that no eps-scaled tolerance can bound.")$(linesearch_reason(status, config)) If rfₐ is accurate enough for you, raise f_abstol above it; otherwise improve the approximation F is built on until its floor lies below the tolerance you need. Set verbosity = 0 to silence this.")
    # A stagnated solve is not an error, but it did not achieve what was asked of it, so say
    # what it *did* achieve and how to make the request attainable. This replaces the former
    # pair of misleading messages (a line-search warning per iteration plus "Solver took 1000
    # iterations."), neither of which named the actual problem.
    # Rate limited because a caller that drives `solve!` in a loop — a time-stepping integrator,
    # say — would otherwise get this once per step for as long as the problem stays unattainable,
    # which is the message flood this replaced. `should_report!` and not `maxlog`: the latter is
    # keyed on the source location, so its budget is process-global *and* never reset, which left a
    # genuinely new failure late in a long run unreported. See `should_report!`.
    (stagnated && config.verbosity ≥ 1 && should_report!(Symbol(:stagnated_, lskey))) &&
        (@warn "Nonlinear solver stagnated after $(status.iterations) iterations: the last $(status.stalls) steps did not move the iterate, so the residual rfₐ = $(status.rfₐ) cannot be reduced further — this is the achievable floor for this problem in this precision. The requested residual tolerance was f_abstol = $(config.f_abstol) (plus f_reltol = $(config.f_reltol) times the initial residual ‖F(x₀)‖).$(linesearch_reason(status, config)) If rfₐ is accurate enough for you, raise f_abstol above it; otherwise rescale F so that its round-off floor lies below the tolerance you need. Set verbosity = 0 to silence this.")
    (status.f_increased && !config.allow_f_increases) &&
        (@warn "The function increased and the solver stopped!")
    (status.rfₐ > config.f_abstol_break) &&
        (@warn "The residual rfₐ has reached the maximally allowed value $(config.f_abstol_break)!")
    (havenonfinite(status) && status.iterations ≥ 1 && config.verbosity ≥ 1) &&
        (@warn "Nonlinear solver encountered NaNs or Infs in solution or function value.")

    # A solve that *converged* despite a line search that kept failing got where it was going, so
    # this is not a warning about the result — but the route it took is worth seeing when you are
    # already looking, and it is the only trace of the line search a solve leaves. `verbosity ≥ 2`,
    # the same gate `LINESEARCH_FLOOR` and `LINESEARCH_STATIONARY` use.
    #
    # `LINESEARCH_FLOOR` does not count towards it, although `linesearch_failures` counts it: the
    # last step of a converged solve reports the merit's round-off floor as a matter of course — a
    # solve that reached its tolerance did so by making the residual as small as the arithmetic
    # allows — so a floor here is the healthy case, and counting it would announce that *every*
    # converged solve did not go smoothly. That is the standard `linesearch_reason` states: one
    # failure is noise from the last step, a count is evidence. It is also what `isbenign`
    # promises — that the tally is named only for a solve that did not converge — and the promise
    # holds for the two messages above, which fire only when it did not.
    # The same outcome is handed to `linesearch_reason`, so the clause names what made the message
    # fire rather than the floor it is deliberately ignoring.
    rough = dominant_linesearch_outcome(status, false)
    (isconverged(status) && !stagnated && !noprogress && config.verbosity ≥ 2 &&
     !isnothing(rough)) &&
        (@info "Nonlinear solver converged after $(status.iterations) iterations to rfₐ = $(status.rfₐ), but not every step went smoothly.$(linesearch_reason(status, config, rough))")

    nothing
end
