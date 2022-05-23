
using Printf

abstract type NonlinearSolver end

config(s::NonlinearSolver) = error("config not implemented for $(typeof(s))")
status(s::NonlinearSolver) = error("status not implemented for $(typeof(s))")
initialize!(s::NonlinearSolver, ::AbstractArray) = error("initialize! not implemented for $(typeof(s))")
solver_step!(s::NonlinearSolver) = error("solver_step! not implemented for $(typeof(s))")


function solve!(x, s::NonlinearSolver)
    initialize!(s, x)

    while !meets_stopping_criteria(status(s), config(s))
        next_iteration!(status(s))
        solver_step!(x, s)
        update!(status(s), x)
        residual!(status(s))
    end

    warn_iteration_number(status(s), config(s))

    return x
end


struct NonlinearSolverException <: Exception
    msg::String
end

Base.showerror(io::IO, e::NonlinearSolverException) = print(io, "Nonlinear Solver Exception: ", e.msg, "!")

# get_solver_status!(solver::NonlinearSolver{T}, status_dict::Dict) where {T} =
#             get_solver_status!(status(solver), params(solver), status_dict)

# get_solver_status(solver::NonlinearSolver{T}) where {T} = get_solver_status!(solver,
#             Dict(:nls_niter => 0,
#                  :nls_atol => zero(T),
#                  :nls_rtol => zero(T),
#                  :nls_stol => zero(T),
#                  :nls_converged => false)
#             )

