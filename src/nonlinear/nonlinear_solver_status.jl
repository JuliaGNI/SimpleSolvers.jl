@doc raw"""
    NonlinearSolverStatus

Stores absolute and successive residuals for `x` and `f`. It is used as a diagnostic tool in [`NewtonSolver`](@ref).

!!! info
    Compare this to the [`NonlinearSolverState`](@ref) and the [`NonlinearSolverCache`](@ref).

# Keys
- `iterations`: number of iterations
- `rxₛ`: successive residual in `x`,
- `rfₐ`: absolute residual in `f`,
- `rfₛ`: successive residual in `f`,
- `x_converged::Bool`
- `f_converged::Bool`
- `f_increased::Bool`

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

    rxₛ::T
    rfₐ::T
    rfₛ::T

    x_converged::Bool
    f_converged::Bool
    f_increased::Bool
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
    x_settled = rxₛ ≤ norm(solution(state)) * config.x_suctol
    f_settled = rfₛ ≤ norm(value(state)) * config.f_suctol

    # The residual counts as small when it passes the standard `atol + rtol·‖F₀‖`
    # residual test with `atol = f_abstol` and `rtol = f_reltol` (relative to the initial
    # residual `‖F(x₀)‖`). This lets a large-magnitude / ill-conditioned solve converge
    # once its residual is reduced by `f_reltol` from `‖F(x₀)‖`, while a step that stalls
    # near `‖F(x₀)‖` still fails. The relative term drops to zero for an uninitialized
    # state (`initial_residual` is `NaN`), leaving the pure absolute `f_abstol` test.
    r₀ = initial_residual(state)
    relative_residual = isnan(r₀) ? zero(rfₐ) : config.f_reltol * r₀
    residual_small = rfₐ ≤ config.f_abstol + relative_residual

    x_converged = x_settled && residual_small
    f_converged = (f_settled && residual_small) || rfₐ ≤ config.f_abstol

    f_increased = norm(value(state)) > norm(previousvalue(state))

    x_converged, f_converged, f_increased
end

function NonlinearSolverStatus(state::NonlinearSolverState{T}, config::Options{T}) where {T}
    rxₛ, rfₐ, rfₛ = residuals(state)
    x_converged, f_converged, f_increased = assess_convergence(rxₛ, rfₐ, rfₛ, config, state)
    NonlinearSolverStatus{T}(iteration_number(state), rxₛ, rfₐ, rfₛ, x_converged, f_converged, f_increased)
end

Base.show(io::IO, status::NonlinearSolverStatus) = print(io,
    (@sprintf "i=%4i" status.iterations), ",\n",
    (@sprintf "rxₛ=%4e" status.rxₛ), ",\n",
    (@sprintf "rfₐ=%4e" status.rfₐ), ",\n",
    (@sprintf "rfₛ=%4e" status.rfₛ))

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
    meets_stopping_criteria(state, config)

Determines whether the iteration stops based on the current [`NonlinearSolverState`](@ref).

!!! warning
    The function `meets_stopping_criteria` may return `true` even if the solver has not converged. To check convergence, call [`assess_convergence`](@ref) (with the same input arguments).

The function `meets_stopping_criteria` returns `true` if one of the following is satisfied:
- the `status::`[`NonlinearSolverStatus`](@ref) is converged (checked with [`isconverged`](@ref)) and `state.iterations ≥ config.min_iterations`,
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
        (status.f_increased && !config.allow_f_increases) ||
        state.iterations ≥ config.max_iterations ||
        status.rfₐ > config.f_abstol_break ||
        (havenan(status) && state.iterations ≥ 1)
end

function nonlinear_solver_warnings(status::NonlinearSolverStatus, config::Options)
    (config.warn_iterations > 0 && status.iterations ≥ config.warn_iterations) && (@warn "Solver took $(status.iterations) iterations.")
    (status.f_increased && !config.allow_f_increases) && (@warn "The function increased and the solver stopped!")
    (status.rfₐ > config.f_abstol_break) && (@warn "The residual rfₐ has reached the maximally allowed value $(config.f_abstol_break)!")
    (havenan(status) && status.iterations ≥ 1 && config.verbosity ≥ 1) && (@warn "Nonlinear solver encountered NaNs in solution or function value.")

    nothing
end
