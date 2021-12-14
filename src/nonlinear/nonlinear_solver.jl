
using Printf

abstract type NonlinearSolver{T} end

solve!(s::NonlinearSolver) = error("solve! not implemented for $(typeof(s))")
status(s::NonlinearSolver) = error("status not implemented for $(typeof(s))")
params(s::NonlinearSolver) = error("params not implemented for $(typeof(s))")

function solve!(s::NonlinearSolver{T}, x₀::Vector{T}) where {T}
    initialize!(s, x₀)
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
        @assert atol ≥ 0
        @assert rtol ≥ 0
        @assert stol ≥ 0
        @assert atol_break > 0
        @assert rtol_break > 0
        @assert stol_break > 0

        new(nmin, nmax, nwarn, atol, rtol, stol, atol^2, rtol^2, stol^2, atol_break, rtol_break, stol_break)
    end
end

function NonlinearSolverParameters(config::Options{T}) where {T}
    NonlinearSolverParameters{T}(
        config.min_iterations,
        config.max_iterations,
        config.warn_iterations,
        config.f_abstol,
        config.f_reltol,
        config.f_reltol,
        config.f_abstol_break,
        config.f_abstol_break,
        config.f_abstol_break
    )
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

function getLinearSolver(x::AbstractVector{T}; linear_solver = :julia) where {T}
    n = length(x)

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
