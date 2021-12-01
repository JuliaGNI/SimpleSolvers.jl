
using Printf

abstract type NonlinearSolver{T} end

solve!(s::NonlinearSolver) = error("solve! not implemented for $(typeof(s))")
status(s::NonlinearSolver) = error("status not implemented for $(typeof(s))")
params(s::NonlinearSolver) = error("params not implemented for $(typeof(s))")

function solve!(s::NonlinearSolver{T}, x₀::Vector{T}) where {T}
    setInitialConditions!(s, x₀)
    solve!(s)
end


struct NonlinearSolverException <: Exception
    msg::String
end

Base.showerror(io::IO, e::NonlinearSolverException) = print(io, "Nonlinear Solver Exception: ", e.msg, "!")


struct NonlinearSolverParameters{T}
    nmin::Int   # minimum number of iterations
    nmax::Int   # maximum number of iterations
    nwarn::Int  # warn if number of iterations is larger than nwarn

    atol::T     # absolute tolerance
    rtol::T     # relative tolerance
    stol::T     # successive tolerance

    atol²::T
    rtol²::T
    stol²::T

    atol_break::T
    rtol_break::T
    stol_break::T

    function NonlinearSolverParameters{T}(nmin, nmax, nwarn, atol, rtol, stol, atol_break, rtol_break, stol_break) where {T}
        @assert nmin ≥ 0
        @assert nmax > 0
        @assert nwarn ≥ 0
        @assert atol > 0
        @assert rtol > 0
        @assert stol > 0
        @assert atol_break > 0
        @assert rtol_break > 0
        @assert stol_break > 0

        new(nmin, nmax, nwarn, atol, rtol, stol, atol^2, rtol^2, stol^2, atol_break, rtol_break, stol_break)
    end
end

function NonlinearSolverParameters(T)
    NonlinearSolverParameters{T}(get_config(:nls_nmin),
                                 get_config(:nls_nmax),
                                 get_config(:nls_nwarn),
                                 get_config(:nls_atol),
                                 get_config(:nls_rtol),
                                 get_config(:nls_stol),
                                 get_config(:nls_atol_break),
                                 get_config(:nls_rtol_break),
                                 get_config(:nls_stol_break))
end


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

function print_solver_status(status::NonlinearSolverStatus, params::NonlinearSolverParameters)
    if (get_config(:verbosity) ≥ 1 && !(check_solver_converged(status, params) && status.i ≤ params.nmax)) ||
        get_config(:verbosity) > 1
        println(status)
    end
end

function check_solver_converged(status::NonlinearSolverStatus, params::NonlinearSolverParameters)
    return status.rₐ ≤ params.atol  ||
           status.rᵣ ≤ params.rtol  ||
           status.rₛ ≤ params.stol
end

function check_solver_status(status::NonlinearSolverStatus, params::NonlinearSolverParameters)
    if any(x -> isnan(x), status.xₚ)
        throw(NonlinearSolverException("Detected NaN"))
    end

    if status.rₐ > params.atol_break
        throw(NonlinearSolverException("Absolute error ($(status.rₐ)) larger than allowed ($(params.atol_break))"))
    end

    if status.rᵣ > params.rtol_break
        throw(NonlinearSolverException("Relative error ($(status.rᵣ)) larger than allowed ($(params.rtol_break))"))
    end

    if status.rₛ > params.stol_break
        throw(NonlinearSolverException("Succesive error ($(status.rₛ)) larger than allowed ($(params.stol_break))"))
    end
end

function get_solver_status!(status::NonlinearSolverStatus, params::NonlinearSolverParameters, status_dict::Dict)
    status_dict[:nls_niter] = status.i
    status_dict[:nls_atol] = status.rₐ
    status_dict[:nls_rtol] = status.rᵣ
    status_dict[:nls_stol] = status.rₛ
    status_dict[:nls_converged] = check_solver_converged(status, params)
    return status_dict
end

get_solver_status!(solver::NonlinearSolver{T}, status_dict::Dict) where {T} =
            get_solver_status!(status(solver), params(solver), status_dict)

get_solver_status(solver::NonlinearSolver{T}) where {T} = get_solver_status!(solver,
            Dict(:nls_niter => 0,
                 :nls_atol => zero(T),
                 :nls_rtol => zero(T),
                 :nls_stol => zero(T),
                 :nls_converged => false)
            )


function getLinearSolver(x::AbstractVector{T}) where {T}
    n = length(x)
    linear_solver = get_config(:ls_solver)

    if linear_solver === nothing || linear_solver == :julia
        linear_solver = LUSolver{T}(n)
    elseif linear_solver == :lapack
        linear_solver = LUSolverLAPACK{T}(BlasInt(n))
    else
        @assert typeof(linear_solver) <: LinearSolver{T}
        @assert n == linear_solver.n
    end
    return linear_solver
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
