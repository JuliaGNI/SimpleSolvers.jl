
mutable struct NonlinearSolverStatus{XT,YT,AXT,AYT}
    i::Int        # iteration number

    rxₐ::XT       # residual (absolute)
    rxᵣ::XT       # residual (relative)
    rfₐ::YT       # residual (absolute)
    rfᵣ::YT       # residual (relative)

    x::AXT        # initial solution
    x̄::AXT        # previous solution
    δ::AXT        # 

    f::AYT        # initial function
    f̄::AYT        # previous function
    γ::AYT        # initial function

    x_converged::Bool
    f_converged::Bool
    g_converged::Bool
    f_increased::Bool

    NonlinearSolverStatus{T}(n) where {T} = new{T,T,Vector{T},Vector{T}}(
        0, 0, 0, 0, 0,
        zeros(T,n), zeros(T,n), zeros(T,n),
        zeros(T,n), zeros(T,n), zeros(T,n),
        false, false, false, false)
end

solution(status::NonlinearSolverStatus) = status.x

function clear!(status::NonlinearSolverStatus{XT,YT}) where {XT,YT}
    status.i = 0

    status.rxₐ = XT(NaN)
    status.rxᵣ = XT(NaN)
    status.rfₐ = YT(NaN)
    status.rfᵣ = YT(NaN)

    status.x̄ .= XT(NaN)
    status.x .= XT(NaN)
    status.δ .= XT(NaN)

    status.f̄ .= YT(NaN)
    status.f .= YT(NaN)
    status.γ .= YT(NaN)

    status.x_converged = false
    status.f_converged = false
    status.f_increased = false
end

Base.show(io::IO, status::NonlinearSolverStatus) = print(io,
                        (@sprintf "    i=%4i" status.i),  ",   ",
                        (@sprintf "rxₐ=%14.8e" status.rxₐ), ",   ",
                        (@sprintf "rxᵣ=%14.8e" status.rxᵣ), ",   ",
                        (@sprintf "rfₐ=%14.8e" status.rfₐ), ",   ",
                        (@sprintf "rfᵣ=%14.8e" status.rfᵣ))

function print_status(status::NonlinearSolverStatus, config::Options)
    if (config.verbosity ≥ 1 && !(assess_convergence!(status, config) && status.i ≤ config.max_iterations)) ||
        config.verbosity > 1
        println(status)
    end
end

increase_iteration_number!(status::NonlinearSolverStatus) = status.i += 1

isconverged(status::NonlinearSolverStatus) = status.x_converged || status.f_converged

function assess_convergence!(status::NonlinearSolverStatus, config::Options)
    x_converged = status.rxₐ ≤ config.x_abstol ||
                  status.rxᵣ ≤ config.x_reltol
    
    f_converged = status.rfₐ ≤ config.f_abstol ||
                  status.rfᵣ ≤ config.f_reltol

    status.x_converged = x_converged
    status.f_converged = f_converged

    status.f_increased = norm(status.f) > norm(status.f̄)

    return isconverged(status)
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
      any(isnan, status.x) ||
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
        println("WARNING: Solver took ", status.i, " iterations.")
    end
end


function residual_absolute!(status::NonlinearSolverStatus{T}, y::Vector{T}) where {T}
    @assert length(y) == length(status.y₀)
    status.rₐ = maxnorm(y)
end

function residual_relative!(status::NonlinearSolverStatus{T}, y::Vector{T}) where {T}
    @assert length(y) == length(status.yₚ)
    status.rᵣ = norm(y .- status.yₚ) / norm(status.yₚ)
end

function residual_successive!(status::NonlinearSolverStatus{T}, δx::Vector{T}, x::Vector{T}) where {T}
    @assert length(δx) == length(x)
    status.rₛ = norm(δx) / norm(x)
end


function residual!(status::NonlinearSolverStatus)
    status.rxₐ = norm(status.δ)
    status.rxᵣ = status.rxₐ / norm(status.x)
    status.rfₐ = norm(status.γ)
    status.rfᵣ = status.rfₐ / norm(status.f)
end


function residual!(status::NonlinearSolverStatus, x, f)
    copyto!(status.x, x)
    copyto!(status.f, f)

    status.δ .= status.x - status.x̄
    status.γ .= status.f - status.f̄

    residual!(status)
end

function initialize!(status::NonlinearSolverStatus, x, f)
    clear!(status)
    status.x̄ .= status.x .= x
    status.f̄ .= status.f .= f
    status.δ .= 0
    status.γ .= 0
end

function next_iteration!(status::NonlinearSolverStatus)
    increase_iteration_number!(status)
    status.x̄ .= status.x
    status.f̄ .= status.f
end
