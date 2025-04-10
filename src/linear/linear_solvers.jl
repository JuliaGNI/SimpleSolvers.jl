
"""
    LinearSolver <: AbstractSolver

A supertype that comprises e.g. [`LUSolver`](@ref) and [`LUSolverLAPACK`](@ref).
"""
abstract type LinearSolver{T} <: AbstractSolver end

factorize!(s::LinearSolver) = error("factorize! not implemented for $(typeof(s))")
LinearAlgebra.ldiv!(s::LinearSolver) = error("ldiv! not implemented for $(typeof(s))")

function LinearSolver(x::AbstractVector{T}; linear_solver = :julia) where {T}
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
