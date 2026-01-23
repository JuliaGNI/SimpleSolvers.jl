@doc raw"""
    NonlinearSolverStatus

Stores absolute, relative and successive residuals for `x` and `f`. It is used as a diagnostic tool in [`NewtonSolver`](@ref).

# Keys
- `i::Int`: iteration number,
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
mutable struct NonlinearSolverStatus{XT,YT,AXT,AYT}
    i::Int

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

    function NonlinearSolverStatus{T}(n::Int) where {T}

        status = new{T,T,Vector{T},Vector{T}}(
            0,
            zero(T), zero(T), # residuals for x
            zero(T), zero(T), # residuals for f
            zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n), # the ics for x
            zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n), # the ics for f
            false, false, false, false)
        clear!(status)
        status
    end

    function NonlinearSolverStatus(x₀::AbstractVector{T}) where {T}
        status = NonlinearSolverStatus{T}(length(x₀))
        initialize!(status, x₀)
        status
    end
end

iteration_number(status) = status.i

"""
    solution(status)

Return the current value of `x` (i.e. the current solution).
"""
solution(status::NonlinearSolverStatus) = status.x

function clear!(status::NonlinearSolverStatus{XT,YT}) where {XT,YT}
    status.i = 0

    status.rxₐ = XT(NaN)
    status.rxₛ = XT(NaN)

    status.rfₐ = YT(NaN)
    status.rfₛ = YT(NaN)

    status.x̄ .= XT(NaN)
    status.x .= XT(NaN)
    status.δ .= XT(NaN)
    status.x̃ .= XT(NaN)

    status.f₀ .= YT(NaN)
    status.f̄ .= YT(NaN)
    status.f .= YT(NaN)
    status.γ .= YT(NaN)

    status.x_converged = false
    status.f_converged = false
    status.f_increased = false

    status
end

Base.show(io::IO, status::NonlinearSolverStatus{XT,YT,AXT,AYT}) where {XT,YT,AXT<:AbstractArray,AYT<:AbstractArray} = print(io,
    (@sprintf "i = %4i" iteration_number(status)), ",\n",
    (@sprintf "x = %4e" status.x[1]), ",\n",
    (@sprintf "f = %4e" status.f[1]), ",\n",
    (@sprintf "rxₐ = %4e" status.rxₐ), ",\n",
    (@sprintf "rxₛ = %4e" status.rxₛ), ",\n",
    (@sprintf "rfₐ = %4e" status.rfₐ), ",\n",
    (@sprintf "rfₛ = %4e" status.rfₛ), ",\n")

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

"""
    increase_iteration_number!(status)

Increase iteration number of `status`.

# Examples

```jldoctest; setup = :(using SimpleSolvers: NonlinearSolverStatus, increase_iteration_number!, iteration_number)
status = NonlinearSolverStatus{Float64}(5)
increase_iteration_number!(status)
iteration_number(status)

# output

1
```
"""
increase_iteration_number!(status::NonlinearSolverStatus) = status.i += 1

isconverged(status::NonlinearSolverStatus) = status.x_converged || status.f_converged

"""
    assess_convergence!(status, config)

Check if one of the following is true for `status::`[`NonlinearSolverStatus`](@ref):
- `status.rxₐ ≤ config.x_abstol`,
- `status.rxₛ ≤ config.x_suctol`,
- `status.rfₐ ≤ config.f_abstol`,
- `status.rfₛ ≤ config.f_suctol`.

Also see [`meets_stopping_criteria`](@ref). The tolerances are by default determined with [`default_tolerance`](@ref).
"""
function assess_convergence!(status::NonlinearSolverStatus, config::Options)
    x_converged = status.rxₛ ≤ config.x_suctol

    f_converged = status.rfₐ ≤ config.f_abstol ||
                  status.rfₛ ≤ config.f_suctol

    status.x_converged = x_converged
    status.f_converged = f_converged

    status.f_increased = norm(status.f) > norm(status.f̄)

    isconverged(status)
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

@doc raw"""
    residual!(status)

Compute the residuals for `status::`[`NonlinearSolverStatus`](@ref).
Note that this does not update `x`, `f`, `δ` or `γ`. These are updated with [`update!(::NonlinearSolverStatus, ::AbstractVector, ::NonlinearProblem)`](@ref).
The computed residuals are the following:
- `rxₐ`: absolute residual in ``x``,
- `rxₛ` : successive residual (the norm of ``\delta``),
- `rfₐ`: absolute residual in ``f``,
- `rfₛ` : successive residual (the norm of ``\gamma``).
"""
function residual!(status::NonlinearSolverStatus)
    status.rxₐ = norm(status.x)
    status.rxₛ = norm(status.δ)
    status.rfₐ = norm(status.f)
    status.rfₛ = norm(status.γ)
    status
end

"""
    initialize!(status, x)

Clear `status::`[`NonlinearSolverStatus`](@ref) (via the function [`clear!`](@ref)).
"""
function initialize!(status::NonlinearSolverStatus, x::AbstractVector)
    clear!(status)
end

"""
    update!(status, x, nls)

Update the `status::`[`NonlinearSolverStatus`](@ref) based on `x` for the [`NonlinearProblem`](@ref) `obj`.

!!! info
   This also updates the problem `nls`!

Sets `x̄` and `f̄` to `x` and `f` respectively and computes `δ` as well as `γ`.
The new `x` and `x̄` stored in `status` are used to compute `δ`.
The new `f` and `f̄` stored in `status` are used to compute `γ`.
See [`NonlinearSolverStatus`](@ref) for an explanation of those variables.
"""
function update!(status::NonlinearSolverStatus, x::AbstractVector, nls::NonlinearProblem, params)
    status.x̄ .= status.x
    status.f̄ .= status.f

    status.x .= x
    value!(status.f, nls, x, params)
    iteration_number(status) == 0 && copy!(status.f₀, status.f)

    status.δ .= status.x .- status.x̄
    status.γ .= status.f .- status.f̄

    residual!(status)

    status
end
