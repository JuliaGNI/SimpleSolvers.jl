@doc raw"""
    NonlinearSolverStatus

Stores absolute and successive residuals for `x` and `f`. It is used as a diagnostic tool in [`NewtonSolver`](@ref).

!!! info
    Compare this to the [`NonlinearSolverState`](@ref) and the [`NonlinearSolverCache`](@ref).

# Keys
- `iterations`: number of iterations
- `stalls`: number of *consecutive* stalled steps, see [`stalled_step`](@ref) and [`isstalled`](@ref),
- `rxₛ`: successive residual in `x`,
- `rfₐ`: absolute residual in `f`,
- `rfₛ`: successive residual in `f`,
- `x_converged::Bool`
- `f_converged::Bool`
- `f_increased::Bool`
- `stalled::Bool`: the *last* step stalled, see [`stalled_step`](@ref)

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

    rxₛ::T
    rfₐ::T
    rfₛ::T

    x_converged::Bool
    f_converged::Bool
    f_increased::Bool
    stalled::Bool
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
function assess_convergence(rxₛ::Number, rfₐ::Number, rfₛ::Number, config::Options, state::NonlinearSolverState)
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
    relative_residual = isnan(r₀) ? zero(rfₐ) : config.f_reltol * r₀
    rfₐ ≤ config.f_abstol + relative_residual
end

"""
    iterate_settled(rxₛ, config, state)

Return `true` when the last step did not move the iterate, `rxₛ ≤ ‖x‖·x_suctol`. Used by
[`assess_convergence`](@ref) and [`stalled_step`](@ref).
"""
iterate_settled(rxₛ::Number, config::Options, state::NonlinearSolverState) =
    rxₛ ≤ norm(solution(state)) * config.x_suctol

@doc raw"""
    stalled_step(rxₛ, rfₐ, config, state)

Return `true` when the last step *stalled*: it left the iterate unchanged (see
[`iterate_settled`](@ref)) while the residual is **not** small (see
[`residual_small`](@ref)).

A stalled step is the failure mode that the residual gate in [`assess_convergence`](@ref)
correctly refuses to call convergence, and that used to be invisible to the solver. The step
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

"""
    record_stall!(state, config)

Update the consecutive-stall counter of `state::`[`NonlinearSolverState`](@ref): increment it
when the last step [`stalled_step`](@ref) *or* the line search flagged a stall (see
[`flag_stall!`](@ref)), and reset it to zero otherwise. The flag is cleared either way.
Returns the new count (see [`stall_number`](@ref)).

This must be called *exactly once per iteration* — [`solve!`](@ref) does so right after
[`update!`](@ref). That is why the counter is not maintained inside
[`assess_convergence`](@ref) or [`NonlinearSolverStatus`](@ref): those are pure and are
evaluated more than once per iteration, so incrementing there would double-count. A
hand-rolled iteration that drives [`solver_step!`](@ref) directly and never calls
`record_stall!` simply keeps the count at zero and behaves exactly as before.
"""
function record_stall!(state::NonlinearSolverState, config::Options)
    rxₛ, rfₐ, _ = residuals(state)
    flagged = state.stallflag
    state.stallflag = false
    # The line-search flag substitutes for `iterate_settled` (it is the same news, one
    # iteration earlier), but it is gated by the *same* residual test: at convergence the
    # merit is also at its round-off floor, and that is success, not stagnation. Keeping the
    # gate is what makes stagnation and convergence mutually exclusive (see `isstalled`).
    stalled = (flagged || iterate_settled(rxₛ, config, state)) && !residual_small(rfₐ, config, state)
    state.stalls = stalled ? state.stalls + 1 : 0
end

function NonlinearSolverStatus(state::NonlinearSolverState{T}, config::Options{T}) where {T}
    rxₛ, rfₐ, rfₛ = residuals(state)
    x_converged, f_converged, f_increased, stalled = assess_convergence(rxₛ, rfₐ, rfₛ, config, state)
    NonlinearSolverStatus{T}(iteration_number(state), stall_number(state), rxₛ, rfₐ, rfₛ, x_converged, f_converged, f_increased, stalled)
end

# The stall line is appended only when it is relevant, so the printout of a fresh status is
# unchanged.
Base.show(io::IO, status::NonlinearSolverStatus) = print(io,
    (@sprintf "i=%4i" status.iterations), ",\n",
    (@sprintf "rxₛ=%4e" status.rxₛ), ",\n",
    (@sprintf "rfₐ=%4e" status.rfₐ), ",\n",
    (@sprintf "rfₛ=%4e" status.rfₛ),
    status.stalls > 0 ? ",\n" * (@sprintf "stalls=%4i" status.stalls) : "")

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
        (isconverged(status) || status.iterations ≥ config.max_iterations || status.iterations ≥ config.warn_iterations)) ||
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
havenan(status::NonlinearSolverStatus) = isnan(status.rxₛ) || isnan(status.rfₐ) || isnan(status.rfₛ)

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
isstalled(status::NonlinearSolverStatus, config::Options) = status.stalls ≥ config.max_stalls

"""
    meets_stopping_criteria(state, config)

Determines whether the iteration stops based on the current [`NonlinearSolverState`](@ref).

!!! warning
    The function `meets_stopping_criteria` may return `true` even if the solver has not converged. To check convergence, call [`assess_convergence`](@ref) (with the same input arguments).

The function `meets_stopping_criteria` returns `true` if one of the following is satisfied:
- the `status::`[`NonlinearSolverStatus`](@ref) is converged (checked with [`isconverged`](@ref)) and `state.iterations ≥ config.min_iterations`,
- the `status` has *stagnated* (checked with [`isstalled`](@ref), i.e. `config.max_stalls` consecutive steps that did not move the iterate while the residual is not small) and `state.iterations ≥ config.min_iterations`,
- `status.f_increased` and `config.allow_f_increases = false` (i.e. `f` increased even though we do not allow it),
- `state.iterations ≥ config.max_iterations`,
- `status.rfₐ > config.f_abstol_break` (by default `Inf`). In theory this returns `true` if the residual gets too big.
- one of the residuals (`rxₛ`, `rfₐ`, `rfₛ`) is `NaN` (checked with `havenan`) and `state.iterations ≥ 1`,
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
        (status.f_increased && !config.allow_f_increases) ||
        state.iterations ≥ config.max_iterations ||
        status.rfₐ > config.f_abstol_break ||
        (havenan(status) && state.iterations ≥ 1)
end

"""
    nonlinear_solver_warnings(status, config)

Report a [`NonlinearSolverStatus`](@ref) at the end of a [`solve!`](@ref): the iteration count
if it reached `warn_iterations`, *stagnation* at the residual floor (see [`isstalled`](@ref)
and [`stalled_step`](@ref)), a disallowed residual increase, a residual beyond
`f_abstol_break`, and `NaN`s. Compare this to [`linesearch_warnings`](@ref), which does the
same for the inner line search, and to [`print_status`](@ref).

All messages except the iteration count and the two hard-failure ones are gated on
`config.verbosity ≥ 1`.
"""
function nonlinear_solver_warnings(status::NonlinearSolverStatus, config::Options)
    (config.warn_iterations > 0 && status.iterations ≥ config.warn_iterations) && (@warn "Solver took $(status.iterations) iterations.")
    # A stagnated solve is not an error, but it did not achieve what was asked of it, so say
    # what it *did* achieve and how to make the request attainable. This replaces the former
    # pair of misleading messages (a line-search warning per iteration plus "Solver took 1000
    # iterations."), neither of which named the actual problem.
    # `maxlog` for the same reason every line-search message has one: a caller that drives
    # `solve!` in a loop — a time-stepping integrator, say — would otherwise get this once per
    # step for as long as the problem stays unattainable, which is the message flood this
    # replaced. Note that `maxlog` is keyed on the source location and so is process-global, not
    # per solve; see `linesearch_warnings`.
    (isstalled(status, config) && config.verbosity ≥ 1) &&
        (@warn "Nonlinear solver stagnated after $(status.iterations) iterations: the last $(status.stalls) steps did not move the iterate, so the residual rfₐ = $(status.rfₐ) cannot be reduced further — this is the achievable floor for this problem in this precision. The requested residual tolerance was f_abstol = $(config.f_abstol) (plus f_reltol = $(config.f_reltol) times the initial residual ‖F(x₀)‖). If rfₐ is accurate enough for you, raise f_abstol above it; otherwise rescale F so that its round-off floor lies below the tolerance you need. Set verbosity = 0 to silence this." maxlog = 3)
    (status.f_increased && !config.allow_f_increases) && (@warn "The function increased and the solver stopped!")
    (status.rfₐ > config.f_abstol_break) && (@warn "The residual rfₐ has reached the maximally allowed value $(config.f_abstol_break)!")
    (havenan(status) && status.iterations ≥ 1 && config.verbosity ≥ 1) && (@warn "Nonlinear solver encountered NaNs in solution or function value.")

    nothing
end
