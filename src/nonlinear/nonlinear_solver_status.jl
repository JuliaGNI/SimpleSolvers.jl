
mutable struct NonlinearSolverStatus{T}
    i::Int         # iteration number
    rₐ::T          # residual (absolute)
    rᵣ::T          # residual (relative)
    rₛ::T          # residual (successive)
    r₀::Vector{T}  # initial residual (absolute)

    x₀::Vector{T}  # initial solution
    xₚ::Vector{T}  # previous solution

    y₀::Vector{T}  # initial function
    yₚ::Vector{T}  # previous function

    NonlinearSolverStatus{T}(n) where {T} = new(0, 0, 0, 0, zeros(T,n),
                                                zeros(T,n), zeros(T,n),
                                                zeros(T,n), zeros(T,n),)
end

Base.show(io::IO, status::NonlinearSolverStatus) = print(io,
                        (@sprintf "    i=%4i" status.i),  ",   ", (@sprintf "rₐ=%14.8e" status.rₐ), ",   ",
                        (@sprintf "rᵣ=%14.8e" status.rᵣ), ",   ", (@sprintf "rₛ=%14.8e" status.rₛ))

function print_solver_status(status::NonlinearSolverStatus, config::Options)
    if (config.verbosity ≥ 1 && !(assess_convergence(status, config) && status.i ≤ config.max_iterations)) ||
        config.verbosity > 1
        println(status)
    end
end

function assess_convergence(status::NonlinearSolverStatus, config::Options)
    return (status.rₐ ≤ config.f_abstol  ||
            status.rᵣ ≤ config.f_reltol) &&
           status.i ≥ config.min_iterations
end

function check_solver_status(status::NonlinearSolverStatus, config::Options)
    if any(x -> isnan(x), status.xₚ)
        throw(NonlinearSolverException("Detected NaN"))
    end

    if status.rₐ > config.f_abstol_break
        throw(NonlinearSolverException("Absolute error ($(status.rₐ)) larger than allowed ($(config.f_abstol_break))"))
    end

    if status.rᵣ > config.f_reltol_break
        throw(NonlinearSolverException("Relative error ($(status.rᵣ)) larger than allowed ($(config.f_reltol_break))"))
    end
end

function get_solver_status!(status::NonlinearSolverStatus, config::Options, status_dict::Dict)
    status_dict[:nls_niter] = status.i
    status_dict[:nls_atol] = status.rₐ
    status_dict[:nls_rtol] = status.rᵣ
    status_dict[:nls_stol] = status.rₛ
    status_dict[:nls_converged] = assess_convergence(status, config)
    return status_dict
end


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

function residual!(status::NonlinearSolverStatus{T}, δx::Vector{T}, x::Vector{T}, y::Vector{T}) where {T}
    residual_absolute!(status, y)
    residual_relative!(status, y)
    residual_successive!(status, δx, x)
    status.xₚ .= x
    status.yₚ .= y
end

function residual_initial!(status::NonlinearSolverStatus{T}, x::Vector{T}, y::Vector{T}) where {T}
    @assert length(x) == length(y) == length(status.r₀)

    status.r₀ .= y.^2
    status.rₐ  = maxnorm(y)
    status.rᵣ  = 1
    status.x₀ .= x
    status.xₚ .= x
    status.y₀ .= y
    status.yₚ .= y
end
