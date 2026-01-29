@doc raw"""
    NonlinearSolverStatus

Stores absolute, relative and successive residuals for `x` and `f`. It is used as a diagnostic tool in [`NewtonSolver`](@ref).

# Keys
- `iteration`: number of iterations
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
    residuals(cache, state)

Compute the residuals for `cache::`[`NonlinearSolverCache`](@ref).
The computed residuals are the following:
- `rxₛ` : successive residual (the norm of ``\delta``),
- `rfₐ`: absolute residual in ``f``,
- `rfₛ` : successive residual (the norm of ``\Delta{}y``).
"""
function residuals(state::NonlinearSolverState)
    rxₛ = l2norm(solution(state), previoussolution(state))
    rfₐ = l2norm(value(state))
    rfₛ = l2norm(value(state), previousvalue(state))

    rxₛ, rfₐ, rfₛ
end

"""
    assess_convergence(rxₛ, rfₐ, rfₛ, config, cache, state)

Check if one of the following is true for `status::`[`NonlinearSolverStatus`](@ref):
- `rxₛ ≤ config.x_suctol`,
- `rfₐ ≤ config.f_abstol`,
- `rfₛ ≤ config.f_suctol`.

Also see [`meets_stopping_criteria`](@ref). The tolerances are by default determined with [`default_tolerance`](@ref).
"""
function assess_convergence(rxₛ::Number, rfₐ::Number, rfₛ::Number, config::Options, state::NonlinearSolverState)
    x_converged = rxₛ ≤ norm(solution(state)) * config.x_suctol

    f_converged = rfₛ ≤ norm(value(state)) * config.f_suctol || rfₐ ≤ config.f_abstol

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
1. The following three are satisfied: (i) `config.verbosity` ``\geq1`` (ii) `assess_convergence!(status, config)` is `false` (iii) `iteration_number(status) > config.max_iterations`
2. `config.verbosity > 1`.
"""
function print_status(status::NonlinearSolverStatus, config::Options)
    if (config.verbosity ≥ 1 && !(isconverged(status) && status.iterations ≤ config.max_iterations)) || config.verbosity > 1
        println(status)
    end
end

isconverged(status::NonlinearSolverStatus) = status.x_converged || status.f_converged
havenan(status::NonlinearSolverStatus) = isnan(status.rxₛ) || isnan(status.rfₐ) || isnan(status.rfₛ)

"""
    meets_stopping_criteria(status, config)

Determines whether the iteration stops based on the current [`NonlinearSolverStatus`](@ref).

!!! warning
    The function `meets_stopping_criteria` may return `true` even if the solver has not converged. To check convergence, call [`assess_convergence`](@ref) (with the same input arguments).

The function `meets_stopping_criteria` returns `true` if one of the following is satisfied:
- the `status::`[`NonlinearSolverStatus`](@ref) is converged (checked with [`assess_convergence`](@ref)) and `iteration_number(status) ≥ config.min_iterations`,
- `status.f_increased` and `config.allow_f_increases = false` (i.e. `f` increased even though we do not allow it),
- `iteration_number(status) ≥ config.max_iterations`,
- if any component in `solution(status)` is `NaN`,
- if any component in `status.f` is `NaN`,
- `status.rxₐ > config.x_abstol_break` (by default `Inf`. In theory this returns `true` if the residual gets too big,
- `status.rfₐ > config.f_abstol_break` (by default `Inf`. In theory this returns `true` if the residual gets too big,
So convergence is only one possible criterion for which [`meets_stopping_criteria`](@ref). We may also satisfy a stopping criterion without having convergence!

# Examples

In the following example we show that `meets_stopping_criteria` evaluates to true when used on a freshly allocated [`NonlinearSolverStatus`](@ref):
```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: NonlinearSolverStatus, meets_stopping_criteria, NonlinearSolverCache, NonlinearSolverState)
config = Options(verbosity=0)
x = [NaN, 2., 3.]
cache = NonlinearSolverCache(x, copy(x))
state = NonlinearSolverState(x)
status = NonlinearSolverStatus(state, config)
meets_stopping_criteria(state, config)

# output

false
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

    # status.x_converged && (@warn "x supposedly converged!")
    # status.f_converged && (@warn "f supposedly converged!")

    nothing
end
