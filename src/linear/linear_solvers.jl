abstract type AbstractLinearSolver <: AbstractSolver end

struct NoLinearSolver <: AbstractLinearSolver end

"""
    LinearSolver <: AbstractSolver

A struct that stores [`LinearSolverMethod`](@ref)s (for example [`LU`](@ref)) and [`LinearSolverCache`](@ref)s (for example [`LUSolverCache`](@ref)). [`LinearSolver`](@ref)s are used to solve [`LinearProblem`](@ref)s.

# Constructors

```julia
LinearSolver(method, cache)
LinearSolver(method, A)
LinearSolver(method, ls::LinearProblem)
LinearSolver(method, x)
```

!!! info
    We note that the constructors do not call the function `factorize`, so only allocate a new matrix. The factorization needs to be done manually.

You can manually factorize by either calling [`factorize!`](@ref) or [`solve!`](@ref).
"""
struct LinearSolver{T,LSMT<:LinearSolverMethod,LSCT<:LinearSolverCache} <: AbstractLinearSolver
    method::LSMT
    cache::LSCT

    LinearSolver(method::LSMT, cache::LSCT) where {T,LSMT<:LinearSolverMethod,LSCT<:LinearSolverCache{T}} = new{T,LSMT,LSCT}(method, cache)
end

function factorize!(lsolver::LinearSolver)
    error("No method `factorize!` implemented for method $(typeof(method(lsolver))).")
end

"""
    cache(ls)

Return the cache of the [`LinearSolver`](@ref).

# Examples

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: cache)
julia> ls = LinearSolver(LU(), [1 2; 3 4]);

julia> cache(ls)
SimpleSolvers.LUSolverCache{Int64, StaticArraysCore.MMatrix{2, 2, Int64, 4}}([1 2; 3 4], [0, 0], [0, 0], 0)
```
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

function LinearSolver(method::LinearSolverMethod, ls::LinearProblem)
    LinearSolver(method, ls.A)
end

function LinearSolver(method::LinearSolverMethod, x::AbstractVector{T}) where {T}
    n = length(x)
    LinearSolver(method, zeros(T, n, n))
end

"""
    solve!(x, ls::LinearSolver, lsys::LinearProblem)

Solve the [`LinearProblem`](@ref) `lsys` with the [`LinearSolver`](@ref) `ls` and store the result in `x`.

Also see [`solve(::LU, ::AbstractMatrix, ::AbstractVector)`](@ref).

# Examples

```jldoctest; setup = :(using SimpleSolvers)
julia> x = zeros(3)
3-element Vector{Float64}:
 0.0
 0.0
 0.0

julia> A = [1.; 0.; 0.;; 0.; 2.; 0.;; 0.; 0.; 4.]
3×3 Matrix{Float64}:
 1.0  0.0  0.0
 0.0  2.0  0.0
 0.0  0.0  4.0

julia> b = ones(3)
3-element Vector{Float64}:
 1.0
 1.0
 1.0

julia> ls = LinearSolver(LU(), x);

julia> problem = LinearProblem(x); update!(problem, A, b);

julia> solve!(x, ls, problem)
3-element Vector{Float64}:
 1.0
 0.5
 0.25

```
"""
function solve!(::AbstractVector, ::LinearSolver, ::LinearProblem)
    error("No method for solve! implemented for this combination of input arguments.")
end

"""
    solve!(ls::LinearSolver, args...)

Solve the [`LinearProblem`](@ref) with the [`LinearSolver`](@ref) `ls`.
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

Comapre this to [`solve(::LinearSolver, ::AbstractVector)`](@ref).
"""
function solve!(::AbstractVector, ::LinearSolver, ::AbstractMatrix, ::AbstractVector)
    error("No method for solve! implemented for this combination of input arguments.")
end
