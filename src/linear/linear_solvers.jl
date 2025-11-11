abstract type AbstractLinearSolver <: AbstractSolver end

struct NoLinearSolver <: AbstractLinearSolver end

"""
    LinearSolver <: AbstractSolver

A struct that stores [`LinearSolverMethod`](@ref)s and [`LinearSolverCache`](@ref)s. [`LinearSolver`](@ref)s are used to solve [`LinearSystem`](@ref)s.

# Constructors

```julia
LinearSolver(method, cache)
LinearSolver(method, A)
LinearSolver(method, ls::LinearSystem)
LinearSolver(method, x)
```

!!! info
    We note that the constructors do not call the function `factorize`, so only allocate a new matrix. The factorization needs to be done manually.

You can manually factorize by either calling [`factorize!`](@ref) or [`solve!`](@ref).
"""
struct LinearSolver{T, LSMT <: LinearSolverMethod, LSCT <: LinearSolverCache} <: AbstractLinearSolver 
    method::LSMT
    cache::LSCT

    LinearSolver(method::LSMT, cache::LSCT) where {T, LSMT <: LinearSolverMethod, LSCT <: LinearSolverCache{T}} = new{T, LSMT, LSCT}(method, cache)
end

"""
    factorize!(lsolver)

Factorize the matrix stored in the [`LinearSolverCache`](@ref) in `lsolver`.

See [`factorize!(::LinearSolver{T, LUT}) where {T, LUT <: LU}`](@ref) for a concrete example.
"""
function factorize!(lsolver::LinearSolver)
    error("No method `factorize!` implemented for method $(typeof(method(lsolver))).")
end

"""
    cache(ls)

Return the cache (of type [`LinearSolverCache`](@ref)) of the [`LinearSolver`](@ref).
"""
cache(ls::LinearSolver) = ls.cache

"""
    method(ls)

Return the method (of type [`LinearSolverMethod`](@ref)) of the [`LinearSolver`](@ref).
"""
method(ls::LinearSolver) = ls.method

LinearAlgebra.ldiv!(::AbstractVector, s::LinearSolver, ::AbstractVector) = error("ldiv! not implemented for $(typeof(s))")

function LinearSolver(method::LinearSolverMethod, A::AbstractArray{T}) where {T}
    cache = LinearSolverCache(method, A)
    LinearSolver(method, cache)
end

function LinearSolver(method::LinearSolverMethod, ls::LinearSystem)
    LinearSolver(method, ls.A)
end

function LinearSolver(method::LinearSolverMethod, x::AbstractVector{T}) where {T}
    n = length(x)
    LinearSolver(method, zeros(T, n, n))
end

"""
    solve!(x, ls::LinearSolver, lsys::LinearSystem)

Solve the [`LinearSystem`](@ref) `lsys` with the [`LinearSolver`](@ref) `ls` and store the result in `x`.
Also see [`solve!(::LinearSolver, ::LinearSystem)`](@ref).
"""
function solve!(::AbstractVector, ::LinearSolver, ::LinearSystem)
    error("No method for solve! implemented for this combination of input arguments.")
end

"""
    solve!(ls::LinearSolver, args...)

Solve the [`LinearSystem`](@ref) with the [`LinearSolver`](@ref) `ls`.
"""
function solve!(::LinearSolver, args...)
    error("No method for solve! implemented for this combination of input arguments $(typeof(args...)).")
end

@doc raw"""
    solve!(x, ls::LinearSolver, b)

Solve the linear system described by:
```math
    Ax = b,
```
and store it in `x`. Here ``b`` is provided as an input argument and the factorized ``A`` is stored in the [`LinearSolver`](@ref) `ls` (respectively its [`LinearSolverCache`](@ref)).
"""
function solve!(::AbstractVector, ::LinearSolver, ::AbstractVector)
    error("No method for solve! implemented for this combination of input arguments.")
end

@doc raw"""
    solve!(x, ls::LinearSolver, A, b)

Solve the linear system described by:
```math
    Ax = b,
```
and store it in `x`. Here ``A`` and ``b`` are provided as an input arguments.

# implementation

Note that, compared to [`solve(::LinearSolver, ::AbstractVector)`](@ref) this method involves an additional *factorization* of `A`.
"""
function solve!(::AbstractVector, ::LinearSolver, ::AbstractMatrix, ::AbstractVector)
    error("No method for solve! implemented for this combination of input arguments.")
end