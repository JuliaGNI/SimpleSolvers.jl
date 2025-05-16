
"""
    LinearSolver <: AbstractSolver

A supertype that comprises e.g. [`LUSolver`](@ref) and [`LUSolverLAPACK`](@ref). These are used to solver [`LinearSystem`](@ref)s.

# Constructor

```julia
LinearSolver(x; linear_solver = :julia)
```
The convenience constructor allocates a specific `struct` derived from `LinearSolver` based on what is supplied to `liner_solver`. The default `:julia` calls the constructor for [`LUSolver`](@ref).
Another option would be `:lapack` which calls [`LUSolverLAPACK`](@ref) and uses the `LinearAlgebra.BLAS` package.
"""
abstract type LinearSolver{T} <: AbstractSolver end

factorize!(s::LinearSolver) = error("factorize! not implemented for $(typeof(s))")
LinearAlgebra.ldiv!(s::LinearSolver) = error("ldiv! not implemented for $(typeof(s))")

function LinearSolver(x::AbstractVector{T}; linearsolver = :julia) where {T}
    n = length(x)

    linear_solver_object = if linearsolver === nothing || linearsolver == :julia
        LUSolver{T}(n)
    elseif linearsolver == :lapack
        LUSolverLAPACK{T}(BlasInt(n))
    else
        @assert typeof(linearsolver) <: LinearSolver{T}
        @assert n == linearsolver.n
        linearsolver
    end

    linear_solver_object
end
