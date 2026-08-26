"""
    NonlinearSolverState <: AbstractSolverState

The `NonlinearSolverState` to be used together with a [`NonlinearSolver`](@ref).

!!! info
    Note the difference to the [`NonlinearSolverCache`](@ref) and the [`NonlinearSolverStatus`](@ref).

# Examples

```jldoctest; setup = :(using SimpleSolvers)
julia> state = NonlinearSolverState(zeros(3))
NonlinearSolverState{Float64, Vector{Float64}, Vector{Float64}}(0, [NaN, NaN, NaN], [NaN, NaN, NaN], [NaN, NaN, NaN], [NaN, NaN, NaN], NaN, 0, false, NaN, 0, [0, 0, 0, 0, 0, 0])
```
"""
mutable struct NonlinearSolverState{T,XT<:AbstractVector{T},YT<:AbstractVector{T}} <: AbstractSolverState
    iterations::Int

    x::XT
    x̄::XT
    y::YT
    ȳ::YT

    r₀::T   # the initial residual ‖F(x₀)‖, set by `initialize!`; reference scale
            # for the relative-residual convergence test (`NaN` until initialized)

    stalls::Int      # number of *consecutive* stalled steps (a step that did not move the
                     # iterate while the residual is not small); maintained by
                     # `record_stall!`, consumed by `meets_stopping_criteria`
    stallflag::Bool  # set by `flag_stall!` when the line search itself reported that the
                     # merit cannot be decreased; OR-ed into the verdict of the next
                     # `record_stall!`, which clears it again

    rf_ref::T                  # the residual as of the last iteration that counted as progress,
    iters_since_progress::Int  # and the number of iterations since; both maintained by
                               # `record_progress!`, the latter read as
                               # `iterations_since_progress` by the report and by `no_progress`.
                               # The counter is *owned* rather than derived from `iterations` so
                               # that, exactly like `stalls`, it stays at zero for an iteration
                               # that never records.

    ls_outcomes::MVector{NLINESEARCH_OUTCOMES,Int}  # how often each `LinesearchOutcome` was
                               # reported during this solve, indexed by `Int(oc) + 1`; maintained
                               # by `record_linesearch!` and read by `NonlinearSolverStatus`

    function NonlinearSolverState(X::AbstractVector{T}, Y::AbstractVector{T}=X) where {T}
        x = zero(X)
        x̄ = zero(X)
        y = zero(Y)
        ȳ = zero(Y)

        x .= T(NaN)
        x̄ .= T(NaN)
        y .= T(NaN)
        ȳ .= T(NaN)

        new{T,typeof(x),typeof(y)}(0, x, x̄, y, ȳ, T(NaN), 0, false, T(NaN), 0,
            zeros(MVector{NLINESEARCH_OUTCOMES,Int}))
    end
end

"""
    iteration_number(state)

Return the number of iterations taken so far, as counted by
[`increase_iteration_number!`](@ref) on `state::`[`NonlinearSolverState`](@ref). Compare this
to [`stall_number`](@ref).
"""
iteration_number(state::NonlinearSolverState) = state.iterations
solution(state::NonlinearSolverState) = state.x
value(state::NonlinearSolverState) = state.y

previoussolution(state::NonlinearSolverState) = state.x̄
previousvalue(state::NonlinearSolverState) = state.ȳ

"""
    initial_residual(state)

Return the initial residual ‖F(x₀)‖ recorded by [`initialize!`](@ref) (`NaN` if the
state has not been initialized). This is the reference scale for the relative-residual
convergence test in [`assess_convergence`](@ref).
"""
initial_residual(state::NonlinearSolverState) = state.r₀

"""
    increase_iteration_number!(state)

To be used together with [`NonlinearSolverState`](@ref).
"""
function increase_iteration_number!(state::NonlinearSolverState)
    state.iterations += 1
end

function NonlinearSolverState{T}(n::Integer, m::Integer=n) where {T}
    x = zeros(T, n)
    y = zeros(T, m)
    NonlinearSolverState(x, y)
end

function initialize!(state::NonlinearSolverState{T}, x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    state.iterations = 0
    state.x .= x
    state.y .= y
    state.r₀ = l2norm(y)   # record the initial residual as the relative-convergence scale
    state.x̄ .= T(NaN)
    state.ȳ .= T(NaN)
    state.stalls = 0
    state.stallflag = false
    # The initial residual is also the first progress reference: a solve is measured against
    # where it started, so `iterations_since_progress` counts from iteration 0.
    state.rf_ref = state.r₀
    state.iters_since_progress = 0
    # Per solve, exactly like `stalls`: what the line search reported during the *previous* solve
    # says nothing about this one, and `nonlinear_solver_warnings` reads the tally to explain the
    # solve that has just finished.
    fill!(state.ls_outcomes, 0)
end

"""
    stall_number(state)

Return the number of *consecutive* stalled steps recorded in
`state::`[`NonlinearSolverState`](@ref) by [`record_stall!`](@ref). Compare this to
[`iteration_number`](@ref).
"""
stall_number(state::NonlinearSolverState) = state.stalls

@doc raw"""
    iterations_since_progress(state)

Return the number of iterations since the residual last dropped by `config.f_stall_factor`, as
recorded in `state::`[`NonlinearSolverState`](@ref) by [`record_progress!`](@ref). Zero on a
freshly initialized state.

This measures the failure mode [`stall_number`](@ref) cannot see. A stalled step is one that did
not move the iterate; here the iterate moves perfectly normally — by far more than the round-off
level of ``x`` — while the residual descends towards a floor above the requested tolerance, so
neither [`stalled_step`](@ref) nor either branch of [`assess_convergence`](@ref) can fire and the
solve spends `max_iterations` in full. [`nonlinear_solver_warnings`](@ref) reports it, and
[`no_progress`](@ref) stops it when the caller has opted in with `f_stall_window`.
"""
iterations_since_progress(state::NonlinearSolverState) = state.iters_since_progress

@doc raw"""
    record_progress!(state, config)
    record_progress!(state, config, rfₐ)

Update the progress reference of `state::`[`NonlinearSolverState`](@ref): when the residual has
dropped to `config.f_stall_factor` times the residual of the last iteration that counted as
progress, that becomes the new reference and the count returned by
[`iterations_since_progress`](@ref) restarts; otherwise the count advances by one. Returns that
count. The three-argument form takes a residual `rfₐ` the caller has already computed; the
two-argument form computes it from `value(state)`.

The reference is therefore monotonically non-increasing — it is the best residual so far, at the
granularity of the factor — which is what makes the measurement immune to a residual that jumps
around: an iteration that undoes the progress of the previous one does not reset the clock. See
[`F_STALL_FACTOR`](@ref) for the choice of granularity.

Like [`record_stall!`](@ref), this is a per-iteration measurement rather than a predicate, so it
must be called exactly once per iteration; [`record_iteration!`](@ref) is what does so, and
carries that contract. That is why the counter lives here and not in
[`NonlinearSolverStatus`](@ref), which is pure and is built more than once per iteration. Because
the count is a field rather than a difference of iteration numbers, a hand-rolled iteration that
drives [`solver_step!`](@ref) directly and never records keeps it at zero and behaves exactly as
before — the same guarantee [`stall_number`](@ref) gives.
"""
function record_progress!(state::NonlinearSolverState, config::Options)
    record_progress!(state, config, l2norm(value(state)))
end

function record_progress!(state::NonlinearSolverState, config::Options, rfₐ::Number)
    # `NaN ≤ x` is false, so a not-yet-initialized reference never counts as progress; it is set
    # by `initialize!` before the first iteration in every solve that goes through `solve!`.
    if rfₐ ≤ config.f_stall_factor * state.rf_ref
        state.rf_ref = rfₐ
        state.iters_since_progress = 0
    else
        state.iters_since_progress += 1
    end
    state.iters_since_progress
end

"""
    linesearch_outcomes(state)

Return the tally of [`LinesearchOutcome`](@ref)s reported during this solve, indexed by
[`linesearch_index`](@ref). Maintained by [`record_linesearch!`](@ref) and reset per solve by
[`initialize!`](@ref); [`NonlinearSolverStatus`](@ref) carries a copy of it out to the caller.
"""
linesearch_outcomes(state::NonlinearSolverState) = state.ls_outcomes

@doc raw"""
    record_linesearch!(state, oc)

Count one [`LinesearchOutcome`](@ref) `oc` into the tally of
`state::`[`NonlinearSolverState`](@ref). Called once per [`solver_step!`](@ref) that consults a
line search, and — like [`record_stall!`](@ref) and [`record_progress!`](@ref) — a measurement
rather than a predicate, so it must be called exactly once per such step.

This is the channel by which a line-search failure reaches the user. A line search reports to its
*caller* through the [`LinesearchStatus`](@ref) that [`solve_with_status`](@ref) returns and emits
nothing; only a caller who invoked [`solve`](@ref) directly is a user, and only that path goes
through [`linesearch_warnings`](@ref). A [`NonlinearSolver`](@ref) is not that caller — it
*consumes* the status ([`flag_stall!`](@ref), and it declines to step along a
`LINESEARCH_NO_DESCENT` direction) and accumulates it here, so that the solve explains itself once,
at the end, through [`nonlinear_solver_warnings`](@ref), instead of once per iteration.

That is not a cosmetic difference. A solve that cannot make progress asks the line search for an
impossible decrease at *every* one of its iterations, so reporting per iteration would turn a
single diagnosis into thousands of identical messages. A `maxlog` cap holds those back only at the
price of being keyed on source location and therefore process-global: once spent it stays spent
for the rest of the session, and a genuine failure in a later solve is silent. Not reporting from
inside the loop removes the flood at its source, so no cap is needed and nothing goes permanently
silent.
"""
function record_linesearch!(state::NonlinearSolverState, oc::LinesearchOutcome)
    state.ls_outcomes[linesearch_index(oc)] += 1
    state
end

"""
    linesearch_failures(state)

The number of steps in this solve whose line search reported a failure, i.e. the tally of
[`linesearch_outcomes`](@ref) summed over the outcomes that are not [`isbenign`](@ref).
"""
linesearch_failures(state::NonlinearSolverState) =
    sum(oc -> isbenign(oc) ? 0 : state.ls_outcomes[linesearch_index(oc)], instances(LinesearchOutcome))

"""
    flag_stall!(state)

Record that the line search of the current step reported that it cannot make progress along
the current direction — either the merit is at its round-off floor ([`isfloor`](@ref)) or the
anchor is not a descent direction at all (`LINESEARCH_NO_DESCENT`, see
[`LinesearchOutcome`](@ref)). The flag is OR-ed into the verdict of the next
[`record_stall!`](@ref), which clears it again, and it makes [`needs_refresh`](@ref) true for
the next step.

This is how a line search that *knows* it cannot help reports one iteration earlier than the
step-based diagnosis of [`stalled_step`](@ref) — which remains the primary mechanism, since it
is the only one that also covers a [`Static`](@ref) step along an underflowed direction, a
collapsed [`DogLegSolver`](@ref) trust-region radius, and a locally expanding
[`PicardSolver`](@ref) map.
"""
function flag_stall!(state::NonlinearSolverState)
    state.stallflag = true
    state
end

"""
    needs_refresh(state)

`true` when the previous step made no progress, i.e. when a stall has been flagged for the
current step ([`flag_stall!`](@ref)) or the consecutive-stall counter is nonzero
([`stall_number`](@ref)).

[`solver_step!`](@ref) passes this to [`maybe_refactorize!`](@ref) as its `stalled` keyword, so
a quasi-Newton solver rebuilds its [`Jacobian`](@ref) immediately after a step that did not
move the iterate instead of waiting for the next `refactorize` multiple. Both sources are
consulted because [`record_stall!`](@ref) consumes the flag into the counter once per
iteration, and a caller who drives [`solver_step!`](@ref) by hand may never call it.
"""
needs_refresh(state::NonlinearSolverState) = state.stallflag || stall_number(state) > 0

"""
    update!(state, x, y)

Update `x̄`, `ȳ`, `x` and `y`.

# Examples

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: NullParameters)
julia> f(y, x, params) = y .= sin.(x .- .5) .^ 2
f (generic function with 1 method)

julia> x = ones(1) / 4
1-element Vector{Float64}:
 0.25

julia> y = zero(x); f(y, x, NullParameters())
1-element Vector{Float64}:
 0.06120871905481365

julia> state = NonlinearSolverState(x)
NonlinearSolverState{Float64, Vector{Float64}, Vector{Float64}}(0, [NaN], [NaN], [NaN], [NaN], NaN, 0, false, NaN, 0, [0, 0, 0, 0, 0, 0])

julia> update!(state, x, y)
NonlinearSolverState{Float64, Vector{Float64}, Vector{Float64}}(0, [0.25], [NaN], [0.06120871905481365], [NaN], NaN, 0, false, NaN, 0, [0, 0, 0, 0, 0, 0])

julia> x = ones(1) / 2
1-element Vector{Float64}:
 0.5

julia> f(y, x, NullParameters())
1-element Vector{Float64}:
 0.0

julia> update!(state, x, y)
NonlinearSolverState{Float64, Vector{Float64}, Vector{Float64}}(0, [0.5], [0.25], [0.0], [0.06120871905481365], NaN, 0, false, NaN, 0, [0, 0, 0, 0, 0, 0])
```

The [`NonlinearSolverState`](@ref) stores the previous solution, the previous value, the current solution and the current value.

All of these are updated during one [`update!`](@ref) step (and initialized with `NaN`s).
"""
function update!(state::NonlinearSolverState{T}, x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    previoussolution(state) .= solution(state)
    previousvalue(state) .= value(state)
    solution(state) .= x
    value(state) .= y

    state
end
