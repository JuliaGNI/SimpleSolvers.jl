@doc raw"""
    NonlinearSolverStatus

Stores absolute, relative and successive residuals for `x` and `f`. It is used as a diagnostic tool in [`NewtonSolver`](@ref).

# Keys
- `i::Int`: iteration number,
- `rxₐ`: absolute residual in `x`,
- `rxᵣ`: relative residual in `x`,
- `rxₛ`: successive residual in `x`,
- `rfₐ`: absolute residual in `f`,
- `rfᵣ`: relative residual in `f`,
- `rfₛ`: successive residual in `f`,
- `x`: the *current solution* (can also be accessed by calling [`solution`](@ref)),
- `x̄`: previous solution
- `δ`: change in solution (see [`direction`](@ref)). This is updated by calling [`update!(::NonlinearSolverStatus, ::AbstractVector, ::MultivariateObjective)`](@ref),
- `x̃`: a variable that gives the *component-wise change* via ``\delta/x``,
- `f₀`: initial function value,
- `f`: current function value,
- `f̄`: previous function value,
- `γ`: records change in `f`. This is updated by calling [`update!(::NonlinearSolverStatus, ::AbstractVector, ::MultivariateObjective)`](@ref),
- `f̃`: variable that gives the change in the function value via ``\gamma/f(x_0)``,
- `x_converged::Bool`
- `f_converged::Bool`
- `g_converged::Bool`
- `f_increased::Bool`

# Examples

```jldoctest; setup = :(using SimpleSolvers: NonlinearSolverStatus)
NonlinearSolverStatus{Float64}(3)

# output

i=   0,
x₁= NaN,
f= NaN,
rxₐ= NaN,
rxᵣ= NaN,
rfₐ= NaN,
rfᵣ= NaN
```

# Extended help

!!! info
    Note the difference between the relative residual for ``x``, `rxᵣ`, and the relative residual for ``f``, `rfᵣ`. For the former we compute
    ```math
    \mathrm{r}x_r = \frac{\mathrm{r}x_a}{||x||},
    ```
    for the latter we compute:
    ```math
    \mathrm{r}f_r = \frac{\mathrm{r}f_a}{||f(x_0)||},
    ```
    i.e. we divide by the initial ``f_0``.
"""
mutable struct NonlinearSolverStatus{XT,YT,AXT,AYT}
    i::Int

    rxₐ::XT
    rxᵣ::XT
    rxₛ::XT

    rfₐ::YT
    rfᵣ::YT
    rfₛ::YT

    x::AXT
    x̄::AXT
    δ::AXT
    x̃::AXT

    f₀::AYT
    f::AYT
    f̄::AYT
    γ::AYT
    f̃::AYT

    x_converged::Bool
    f_converged::Bool
    g_converged::Bool
    f_increased::Bool

    function NonlinearSolverStatus{T}(n::Int) where {T}
        
        status = new{T,T,Vector{T},Vector{T}}(
        0, 
        zero(T), zero(T), zero(T), # residuals for x
        zero(T), zero(T), zero(T), # residuals for f
        zeros(T,n), zeros(T,n), zeros(T,n), zeros(T,n), # the ics for x
        zeros(T, n), zeros(T,n), zeros(T,n), zeros(T,n), zeros(T,n), # the ics for f
        false, false, false, false)
        clear!(status)
    end
end

"""
    solution(status)

Return the current value of `x` (i.e. the current solution).
"""
solution(status::NonlinearSolverStatus) = status.x

function clear!(status::NonlinearSolverStatus{XT,YT}) where {XT,YT}
    status.i = 0

    status.rxₐ = XT(NaN)
    status.rxᵣ = XT(NaN)
    status.rxₛ = XT(NaN)

    status.rfₐ = YT(NaN)
    status.rfᵣ = YT(NaN)
    status.rfₛ = YT(NaN)

    status.x̄ .= alloc_x(status.x)
    status.x .= alloc_x(status.x)
    status.δ .= alloc_x(status.x)
    status.x̃ .= alloc_x(status.x)

    status.f̄ .= alloc_f(status.f)
    status.f .= alloc_f(status.f)
    status.γ .= alloc_f(status.f)
    status.f̃ .= alloc_f(status.f)

    status.x_converged = false
    status.f_converged = false
    status.f_increased = false

    status
end

Base.show(io::IO, status::NonlinearSolverStatus{XT,YT,AXT}) where {XT, YT, AXT <: Number} = print(io,
                        (@sprintf "i=%4i" status.i),  ",\n",
                        (@sprintf "x=%4i" status.x),  ",\n",
                        (@sprintf "f=%4i" status.f),  ",\n",
                        (@sprintf "rxₐ=%4e" status.rxₐ), ",\n",
                        (@sprintf "rxᵣ=%4e" status.rxᵣ), ",\n",
                        (@sprintf "rfₐ=%4e" status.rfₐ), ",\n",
                        (@sprintf "rfᵣ=%4e" status.rfᵣ))

Base.show(io::IO, status::NonlinearSolverStatus{XT,YT,AXT, AYT}) where {XT, YT, AXT <: AbstractArray, AYT <: Number} = print(io,
                        (@sprintf "i=%4i" status.i),  ",\n",
                        (@sprintf "x₁=%4i" status.x[1]),  ",\n",
                        (@sprintf "f=%4i" status.f),  ",\n",
                        (@sprintf "rxₐ=%4e" status.rxₐ), ",\n",
                        (@sprintf "rxᵣ=%4e" status.rxᵣ), ",\n",
                        (@sprintf "rfₐ=%4e" status.rfₐ), ",\n",
                        (@sprintf "rfᵣ=%4e" status.rfᵣ))

Base.show(io::IO, status::NonlinearSolverStatus{XT,YT,AXT, AYT}) where {XT, YT, AXT <: AbstractArray, AYT <: AbstractArray} = print(io,
                        (@sprintf "i=%4i" status.i),  ",\n",
                        (@sprintf "x₁=%4i" status.x[1]),  ",\n",
                        (@sprintf "f=%4i" status.f[1]),  ",\n",
                        (@sprintf "rxₐ=%4e" status.rxₐ), ",\n",
                        (@sprintf "rxᵣ=%4e" status.rxᵣ), ",\n",
                        (@sprintf "rfₐ=%4e" status.rfₐ), ",\n",
                        (@sprintf "rfᵣ=%4e" status.rfᵣ))

@doc raw"""
    print_status(status, config)

Print the solver status if:
1. The following three are satisfied: (i) `config.verbosity` ``\geq1`` (ii) `assess_convergence!(status, config)` is `false` (iii) `status.i > config.max_iterations`
2. `config.verbosity > 1`.
"""
function print_status(status::NonlinearSolverStatus, config::Options)
    if (config.verbosity ≥ 1 && !(assess_convergence!(status, config) && status.i ≤ config.max_iterations)) ||
        config.verbosity > 1
        println(status)
    end
end

"""
    increase_iteration_number!(status)

Increase iteration number of `status`.

# Examples

```jldoctest; setup = :(using SimpleSolvers: NonlinearSolverStatus, increase_iteration_number!)
status = NonlinearSolverStatus{Float64}(5)
increase_iteration_number!(status)
status.i

# output

1
```
"""
increase_iteration_number!(status::NonlinearSolverStatus) = status.i += 1

isconverged(status::NonlinearSolverStatus) = status.x_converged || status.f_converged

"""
    assess_convergence!(status, config)

Check if one of the following is true for `status::`[`NonlinearSolverStatus`](@ref):
- `status.rxₐ ≤ config.x_abstol`. See [`X_ABSTOL`](@ref),
- `status.rxᵣ ≤ config.x_reltol`. See [`X_RELTOL`](@ref),
- `status.rxₛ ≤ config.x_suctol`. See [`X_SUCTOL`](@ref),
- `status.rfₐ ≤ config.f_abstol`. See [`F_ABSTOL`](@ref),
- `status.rfᵣ ≤ config.f_reltol`. See [`F_RELTOL`](@ref),
- `status.rfₛ ≤ config.f_suctol`. See [`F_SUCTOL`](@ref).

Also see [`meets_stopping_criteria`](@ref).
"""
function assess_convergence!(status::NonlinearSolverStatus, config::Options)
    x_converged = status.rxₐ ≤ config.x_abstol ||
                  status.rxᵣ ≤ config.x_reltol ||
                  status.rxₛ ≤ config.x_suctol
    
    f_converged = status.rfₐ ≤ config.f_abstol ||
                  status.rfᵣ ≤ config.f_reltol ||
                  status.rfₛ ≤ config.f_suctol

    status.x_converged = x_converged
    status.f_converged = f_converged

    status.f_increased = norm(status.f) > norm(status.f̄)

    isconverged(status)
end

"""
    meets_stopping_criteria(status, config)

Determines whether the iteration stops based on the current [`NonlinearSolverStatus`](@ref).

!!! warn
    The function `meets_stopping_criteria` may return `true` even if the solver has not converged. To check convergence, call `assess_convergence!` (with the same input arguments).

The function `meets_stopping_criteria` returns `true` if one of the following is satisfied:
- the `status::`[`NonlinearSolverStatus`](@ref) is converged (checked with [`assess_convergence!`](@ref)) and `status.i ≥ config.min_iterations`,
- `status.f_increased` and `config.allow_f_increases = false` (i.e. `f` increased even though we do not allow it),
- `status.i ≥ config.max_iterations`, 
- if any component in `solution(status)` is `NaN`,
- if any component in `status.f` is `NaN`,
- `status.rxₐ > config.x_abstol_break` (see [`X_ABSTOL_BREAK`](@ref)). In theory this returns `true` if the residual gets too big, but the default [`X_ABSTOL_BREAK`](@ref) is $(X_ABSTOL_BREAK),
- `status.rxᵣ > config.x_reltol_break` (see [`X_RELTOL_BREAK`](@ref)). In theory this returns `true` if the residual gets too big, but the default [`X_RELTOL_BREAK`](@ref) is $(X_RELTOL_BREAK), 
- `status.rfₐ > config.f_abstol_break` (see [`F_ABSTOL_BREAK`](@ref)). In theory this returns `true` if the residual gets too big, but the default [`F_ABSTOL_BREAK`](@ref) is $(F_ABSTOL_BREAK),
- `status.rfᵣ > config.f_reltol_break` (see [`F_RELTOL_BREAK`](@ref)). In theory this returns `true` if the residual gets too big, but the default [`F_RELTOL_BREAK`](@ref) is $(F_RELTOL_BREAK).
So convergence is only one possible criterion for which [`meets_stopping_criteria`](@ref). We may also satisfy a stopping criterion without having convergence!

# Examples

In the following example we show that `meets_stopping_criteria` evaluates to true when used on a freshly allocated [`NonlinearSolverStatus`](@ref):
```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: NonlinearSolverStatus, meets_stopping_criteria)
status = NonlinearSolverStatus{Float64}(5)
config = Options()
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

    ( isconverged(status) && status.i ≥ config.min_iterations ) ||
    ( status.f_increased && !config.allow_f_increases ) ||
      status.i ≥ config.max_iterations ||
      status.rxₐ > config.x_abstol_break ||
      status.rxᵣ > config.x_reltol_break ||
      status.rfₐ > config.f_abstol_break ||
      status.rfᵣ > config.f_reltol_break ||
      any(isnan, solution(status)) ||
      any(isnan, status.f)
end

function warn_iteration_number(status::NonlinearSolverStatus, config::Options)
    if config.warn_iterations > 0 && status.i ≥ config.warn_iterations
        @warn "Solver took $(status.i) iterations."
    end
    nothing
end

@doc raw"""
    residual!(status)

Compute the residuals for `status::`[`NonlinearSolverStatus`](@ref).
Note that this does not update `x`, `f`, `δ` or `γ`. These are updated with [`update!(::NonlinearSolverStatus, ::AbstractVector, ::MultivariateObjective)`](@ref).
The computed residuals are the following:
- `rxₐ`: absolute residual in ``x``,
- `rxᵣ`: relative residual in ``x``,
- `rxₛ` : successive residual (the norm of ``\delta``),
- `rfₐ`: absolute residual in ``f``,
- `rfᵣ`: relative residual in ``f``,
- `rfₛ` : successive residual (the norm of ``\gamma``).
"""
function residual!(status::NonlinearSolverStatus)
    status.rxₐ = norm(status.δ)
    status.rxᵣ = status.rxₐ / norm(status.x)
    status.x̃ .= status.δ ./ status.x
    status.rxₛ = norm(status.δ)

    status.rfₐ = norm(status.f)
    status.rfᵣ = norm(status.f) / norm(status.f₀)
    status.f̃ .= status.γ ./ status.f
    status.rfₛ = norm(status.γ)

    nothing
end

"""
    initialize!(status, x, f)

Clear `status::`[`NonlinearSolverStatus`](@ref) (via the function [`clear!`](@ref)) and initialize it based on `x` and the function `f`.
"""
function initialize!(status::NonlinearSolverStatus, x::AbstractVector, obj::AbstractObjective)
    clear!(status)
    clear!(obj)
    copyto!(solution(status), x)
    value!(obj, x)
    status.f .= value(obj)
    status.f₀ .= copy(status.f)
    
    status
end

"""
    update!(status, x, obj)

Update the `status::`[`NonlinearSolverStatus`](@ref) based on `x` for the [`MultivariateObjective`](@ref) `obj`.

The new `x` and `x̄` stored in `status` are used to compute `δ`.
The new `f` and `f̄` stored in `status` are used to compute `γ`.
See [`NonlinearSolverStatus`](@ref) for an explanation of those variables.
"""
function update!(status::NonlinearSolverStatus, x::AbstractVector, obj::MultivariateObjective)
    copyto!(solution(status), x)
    value!(obj, x)
    status.f .= value(obj)

    status.δ .= solution(status) .- status.x̄
    status.γ .= status.f .- status.f̄
    
    status
end

"""
    next_iteration!(status)

Call [`increase_iteration_number!`](@ref), set `x̄` and `f̄` to `x` and `f` respectively and `δ` as well as `γ` to 0.
"""
function next_iteration!(status::NonlinearSolverStatus)
    increase_iteration_number!(status)
    status.x̄ .= solution(status)
    status.f̄ .= status.f
    status.δ .= 0
    status.γ .= 0
    
    status
end