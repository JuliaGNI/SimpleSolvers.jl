using Printf

"""
    NonlinearSolver <: AbstractSolver

A supertype that comprises e.g. [`NewtonSolver`](@ref).
"""
abstract type NonlinearSolver <: AbstractSolver end

config(s::NonlinearSolver) = error("config not implemented for $(typeof(s))")
status(s::NonlinearSolver) = error("status not implemented for $(typeof(s))")
initialize!(s::NonlinearSolver, ::AbstractArray) = error("initialize! not implemented for $(typeof(s))")
solver_step!(s::NonlinearSolver) = error("solver_step! not implemented for $(typeof(s))")

struct NonlinearSolverException <: Exception
    msg::String
end

Base.showerror(io::IO, e::NonlinearSolverException) = print(io, "Nonlinear Solver Exception: ", e.msg, "!")