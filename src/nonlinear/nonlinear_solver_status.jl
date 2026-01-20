@doc raw"""
    NonlinearSolverStatus

Stores absolute, relative and successive residuals for `x` and `f`. It is used as a diagnostic tool in [`NewtonSolver`](@ref).

# Keys
- `rxₐ`: absolute residual in `x`,
- `rxₛ`: successive residual in `x`,
- `rfₐ`: absolute residual in `f`,
- `rfₛ`: successive residual in `f`,
- `x`: the *current solution* (can also be accessed by calling [`solution`](@ref)),
- `x̄`: previous solution
- `δ`: change in solution (see [`direction`](@ref)). This is updated by calling [`update!(::NonlinearSolverStatus, ::AbstractVector, ::NonlinearProblem)`](@ref),
- `x̃`: a variable that gives the *component-wise change* via ``\delta/x``,
- `f₀`: initial function value,
- `f`: current function value,
- `f̄`: previous function value,
- `γ`: records change in `f`. This is updated by calling [`update!(::NonlinearSolverStatus, ::AbstractVector, ::NonlinearProblem)`](@ref),
- `x_converged::Bool`
- `f_converged::Bool`
- `g_converged::Bool`
- `f_increased::Bool`

# Examples

```jldoctest; setup = :(using SimpleSolvers: NonlinearSolverStatus)
NonlinearSolverStatus{Float64}(3)

# output

i=   0,
x= NaN,
f= NaN,
rxₐ= NaN,
rfₐ= NaN
```
"""
struct NonlinearSolverStatus{XT,YT,AXT,AYT}
    rxₐ::XT
    rxₛ::XT

    rfₐ::YT
    rfₛ::YT

    x::AXT
    x̄::AXT
    δ::AXT
    x̃::AXT

    f₀::AYT
    f::AYT
    f̄::AYT
    γ::AYT

    x_converged::Bool
    f_converged::Bool
    g_converged::Bool
    f_increased::Bool
end

@doc raw"""
    residual(status)

Compute the residuals for `status::`[`NonlinearSolverStatus`](@ref).
Note that this does not update `x`, `f`, `δ` or `γ`. These are updated with [`update!(::NonlinearSolverStatus, ::AbstractVector, ::NonlinearProblem)`](@ref).
The computed residuals are the following:
- `rxₐ`: absolute residual in ``x``,
- `rxₛ` : successive residual (the norm of ``\delta``),
- `rfₐ`: absolute residual in ``f``,
- `rfₛ` : successive residual (the norm of ``\gamma``).
"""
function residual(status::NonlinearSolverStatus, cache::NonlinearSolverCache)
    rxₐ = norm(direction(cache))
    rxₛ = norm(direction(cache))

    rfₐ = norm(value(cache))
    rfₛ = norm(cache.γ)

    rxₐ, rxₛ, rfₐ, rfₛ
end

function NonlinearSolverStatus(state::NonlinearSolverState{T}, cache::NonlinearSolverCache{T}, config::Options{T}, params) where {T}
    residual!
end

"""
    solution(status)

Return the current value of `x` (i.e. the current solution).
"""
solution(status::NonlinearSolverStatus) = status.x

Base.show(io::IO, status::NonlinearSolverStatus{XT,YT,AXT,AYT}) where {XT,YT,AXT<:AbstractArray,AYT<:AbstractArray} = print(io,
    (@sprintf "x=%4e" status.x[1]), ",\n",
    (@sprintf "f=%4e" status.f[1]), ",\n",
    (@sprintf "rxₐ=%4e" status.rxₐ), ",\n",
    (@sprintf "rfₐ=%4e" status.rfₐ))

@doc raw"""
    print_status(status, config)

Print the solver status if:
1. The following three are satisfied: (i) `config.verbosity` ``\geq1`` (ii) `assess_convergence!(status, config)` is `false` (iii) `iteration_number(status) > config.max_iterations`
2. `config.verbosity > 1`.
"""
function print_status(status::NonlinearSolverStatus, config::Options)
    if (config.verbosity ≥ 1 && !(assess_convergence!(status, config) && iteration_number(status) ≤ config.max_iterations)) ||
       config.verbosity > 1
        println(status)
    end
end

isconverged(status::NonlinearSolverStatus) = status.x_converged || status.f_converged

"""
    assess_convergence(status, config)

Check if one of the following is true for `status::`[`NonlinearSolverStatus`](@ref):
- `status.rxₐ ≤ config.x_abstol`,
- `status.rxₛ ≤ config.x_suctol`,
- `status.rfₐ ≤ config.f_abstol`,
- `status.rfₛ ≤ config.f_suctol`.

Also see [`meets_stopping_criteria`](@ref). The tolerances are by default determined with [`default_tolerance`](@ref).
"""
function assess_convergence(status::NonlinearSolverStatus, config::Options, f, f̄)
    x_converged = status.rxₛ ≤ config.x_suctol

    f_converged = status.rfₐ ≤ config.f_abstol ||
                  status.rfₛ ≤ config.f_suctol

    f_increased = norm(f) > norm(f̄)

    x_converged, f_converged, f_increased
end

"""
    meets_stopping_criteria(status, config)

Determines whether the iteration stops based on the current [`NonlinearSolverStatus`](@ref).

!!! warning
    The function `meets_stopping_criteria` may return `true` even if the solver has not converged. To check convergence, call `assess_convergence!` (with the same input arguments).

The function `meets_stopping_criteria` returns `true` if one of the following is satisfied:
- the `status::`[`NonlinearSolverStatus`](@ref) is converged (checked with [`assess_convergence!`](@ref)) and `iteration_number(status) ≥ config.min_iterations`,
- `status.f_increased` and `config.allow_f_increases = false` (i.e. `f` increased even though we do not allow it),
- `iteration_number(status) ≥ config.max_iterations`,
- if any component in `solution(status)` is `NaN`,
- if any component in `status.f` is `NaN`,
- `status.rxₐ > config.x_abstol_break` (by default `Inf`. In theory this returns `true` if the residual gets too big,
- `status.rfₐ > config.f_abstol_break` (by default `Inf`. In theory this returns `true` if the residual gets too big,
So convergence is only one possible criterion for which [`meets_stopping_criteria`](@ref). We may also satisfy a stopping criterion without having convergence!

# Examples

In the following example we show that `meets_stopping_criteria` evaluates to true when used on a freshly allocated [`NonlinearSolverStatus`](@ref):
```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: NonlinearSolverStatus, meets_stopping_criteria)
status = NonlinearSolverStatus{Float64}(5)
config = Options(verbosity=0)
meets_stopping_criteria(status, config)

# output

true
```
This obviously has not converged. To check convergence we can use [`assess_convergence!`](@ref):
```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: NonlinearSolverStatus, assess_convergence!)
status = NonlinearSolverStatus{Float64}(5)
config = Options()
assess_convergence!(status, config)

# output

false
```
"""
function meets_stopping_criteria(status::NonlinearSolverStatus, config::Options)
    assess_convergence!(status, config)

    havenan = any(isnan, solution(status)) || any(isnan, status.f)

    (havenan && config.verbosity ≥ 1) && (@warn "Nonlinear solver encountered NaNs in solution or function value.")

    (isconverged(status) && iteration_number(status) ≥ config.min_iterations) ||
        (status.f_increased && !config.allow_f_increases) ||
        iteration_number(status) ≥ config.max_iterations ||
        status.rxₐ > config.x_abstol_break ||
        status.rfₐ > config.f_abstol_break ||
        havenan
end

function warn_iteration_number(status::NonlinearSolverStatus, config::Options)
    if config.warn_iterations > 0 && iteration_number(status) ≥ config.warn_iterations
        @warn "Solver took $(iteration_number(status)) iterations."
    end
    nothing
end