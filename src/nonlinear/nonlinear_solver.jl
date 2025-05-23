using Printf

"""
    NonlinearSolver <: AbstractSolver

A supertype that comprises e.g. [`AbstractNewtonSolver`](@ref).
"""
abstract type NonlinearSolver <: AbstractSolver end

config(s::NonlinearSolver) = error("config not implemented for $(typeof(s))")
status(s::NonlinearSolver) = error("status not implemented for $(typeof(s))")
initialize!(s::NonlinearSolver, ::AbstractArray) = error("initialize! not implemented for $(typeof(s))")
solver_step!(s::NonlinearSolver) = error("solver_step! not implemented for $(typeof(s))")

function solve!(x::AbstractArray, obj::AbstractObjective, jacobian!, s::NonlinearSolver)
    initialize!(s, x, obj)

    while !meets_stopping_criteria(status(s), config(s))
        next_iteration!(status(s))
        solver_step!(x, obj, jacobian!, s)
        update!(status(s), x, obj)
        residual!(status(s))
    end

    warn_iteration_number(status(s), config(s))

    x
end

solve!(x::AbstractArray, f::Callable, jacobian!, s::NonlinearSolver) = solve!(x, MultivariateObjective(f, x), jacobian!, s)

# solve!(x, f, s::NonlinearSolver) = solve!(x, f, f, s)

struct NonlinearSolverException <: Exception
    msg::String
end

Base.showerror(io::IO, e::NonlinearSolverException) = print(io, "Nonlinear Solver Exception: ", e.msg, "!")

# (solver::NonlinearSolver{T}, status_dict::Dict) where {T} =
#             get_solver_status!(status(solver), params(solver), status_dict)

# get_solver_status(solver::NonlinearSolver{T}) where {T} = get_solver_status!(solver,
#             Dict(:nls_niter => 0,
#                  :nls_atol => zero(T),
#                  :nls_rtol => zero(T),
#                  :nls_stol => zero(T),
#                  :nls_converged => false)
#             )