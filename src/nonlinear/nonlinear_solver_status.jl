"""
    NonlinearSolverStatus

Stores absolute, relative and successive residuals for `x` and `f`.

# Keys
- `i::Int`: iteration number,
- `rxₐ`: absolute residual in `x`,
- `rxᵣ`: relative residual in `x`,
- `rxₛ`: successive residual in `x`,
- `rfₐ`: absolute residual in `f`,
- `rfᵣ`: relative residual in `f`,
- `rfₛ`: successive residual in `f`,
- `x`: initial solution,
- `x̄`: previous solution
- `δ`: change in solution (see [`direction`](@ref)),
- `x̃`: temporary variable similar to `x`,
- `f`: initial function value,
- `f̄`: previous function value,
- `γ`: records change in `f`
- `f̃`: temporary variable similar to `f`
- `x_converged::Bool`
- `f_converged::Bool`
- `g_converged::Bool`
- `f_increased::Bool`

# Examples

```jldoctest; setup = :(using SimpleSolvers: NonlinearSolverStatus)
NonlinearSolverStatus{Float64}(3)

# output

i=   0,
rxₐ=0.00000000e+00,
rxᵣ=0.00000000e+00,
rfₐ=0.00000000e+00,
rfᵣ=0.00000000e+00
```
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

    f::AYT
    f̄::AYT
    γ::AYT
    f̃::AYT

    x_converged::Bool
    f_converged::Bool
    g_converged::Bool
    f_increased::Bool

    NonlinearSolverStatus{T}(n::Int) where {T} = new{T,T,Vector{T},Vector{T}}(
        0, 
        zero(T), zero(T), zero(T), # residuals for x
        zero(T), zero(T), zero(T), # residuals for f
        zeros(T,n), zeros(T,n), zeros(T,n), zeros(T,n), # the ics for x
        zeros(T,n), zeros(T,n), zeros(T,n), zeros(T,n), # the ics for f
        false, false, false, false)
end

solution(status::NonlinearSolverStatus) = status.x

# why do we initialize with zero but clear to NaN??
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
end

Base.show(io::IO, status::NonlinearSolverStatus) = print(io,
                        (@sprintf "i=%4i" status.i),  ",\n",
                        (@sprintf "rxₐ=%14.8e" status.rxₐ), ",\n",
                        (@sprintf "rxᵣ=%14.8e" status.rxᵣ), ",\n",
                        (@sprintf "rfₐ=%14.8e" status.rfₐ), ",\n",
                        (@sprintf "rfᵣ=%14.8e" status.rfᵣ))

@doc raw"""
    print_status(status, config)

Print the solver staus if:
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

# function check_solver_status(status::NonlinearSolverStatus, config::Options)
#     if any(x -> isnan(x), status.xₚ)
#         throw(NonlinearSolverException("Detected NaN"))
#     end

#     if status.rₐ > config.f_abstol_break
#         throw(NonlinearSolverException("Absolute error ($(status.rₐ)) larger than allowed ($(config.f_abstol_break))"))
#     end

#     if status.rᵣ > config.f_reltol_break
#         throw(NonlinearSolverException("Relative error ($(status.rᵣ)) larger than allowed ($(config.f_reltol_break))"))
#     end
# end

# function get_solver_status!(status::NonlinearSolverStatus, config::Options, status_dict::Dict)
#     status_dict[:nls_niter] = status.i
#     status_dict[:nls_atol] = status.rₐ
#     status_dict[:nls_rtol] = status.rᵣ
#     status_dict[:nls_stol] = status.rₛ
#     status_dict[:nls_converged] = assess_convergence(status, config)
#     return status_dict
# end

function warn_iteration_number(status::NonlinearSolverStatus, config::Options)
    if config.warn_iterations > 0 && status.i ≥ config.warn_iterations
        @warn "Solver took $(status.i) iterations."
    end
    nothing
end

function residual!(status::NonlinearSolverStatus)
    status.rxₐ = norm(status.δ)
    status.rxᵣ = status.rxₐ / norm(status.x)
    status.x̃ .= status.δ ./ status.x
    status.rxₛ = norm(status.δ)

    status.rfₐ = norm(status.f)
    @warn "In order to compute the relative residual for f, we have to store f₀, the initial f. This is not done at the moment."
    status.rfᵣ = norm(status.f) / norm(status.f̃)
    status.f̃ .= status.γ ./ status.f
    status.rfₛ = norm(status.γ)

    nothing
end

"""
    initialize!(status, x, f)

Clear `status` and initialize it based on `x` and the function `f`.
"""
function initialize!(status::NonlinearSolverStatus, x::AbstractVector, obj::AbstractObjective)
    clear!(status)
    clear!(obj)
    copyto!(solution(status), x)
    value!(obj, x)
    status.f .= value(obj)
    
    status
end

"""
    update!(status, x, f)

Update `status` based on `x` and the function `f`.

The new `x` and `x̄` stored in `status` are used to compute `δ`.
The new `f` and `f̄` stored in `status` are used to compute `γ`.
"""
function update!(status::NonlinearSolverStatus, x::AbstractVector, f::Callable)
    copyto!(solution(status), x)
    f(status.f, x)

    status.δ .= solution(status) .- status.x̄
    status.γ .= status.f .- status.f̄
    
    status
end

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