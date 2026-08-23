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

For the default `LU()`, a small matrix (leading dimension ≤ [`N_STATIC_THRESHOLD`](@ref)) is stored as a mutable static matrix (`MMatrix`):

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: cache)
julia> ls = LinearSolver(LU(), [1.0 2.0; 3.0 4.0]);

julia> cache(ls)
SimpleSolvers.LUSolverCache{Float64, StaticArraysCore.MMatrix{2, 2, Float64, 4}}([1.0 2.0; 3.0 4.0], [0, 0], [0, 0], 0)
```

Passing `static=false` forces a plain `Matrix` cache regardless of size:

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: cache)
julia> ls = LinearSolver(LU(; static=false), [1.0 2.0; 3.0 4.0]);

julia> cache(ls)
SimpleSolvers.LUSolverCache{Float64, Matrix{Float64}}([1.0 2.0; 3.0 4.0], [0, 0], [0, 0], 0)
```
"""
cache(ls::LinearSolver) = ls.cache

"""
    method(ls)

Return the method (of type [`LinearSolverMethod`](@ref)) of the [`LinearSolver`](@ref).
"""
method(ls::LinearSolver) = ls.method

"""
    singular_index(lsolver)

Return the index of the first zero pivot encountered by [`factorize!`](@ref), or `0` if the
factorization succeeded.

This is the one piece of factorization state that callers outside the linear solver need:
[`ldiv!`](@ref) turns a non-zero index into a `SingularException`, and the
[`DogLegSolver`](@ref) reads it to decide whether the Newton leg of the step is available at
all (see [`SimpleSolvers.directions!`](@ref)). Every [`LinearSolverMethod`](@ref) therefore
has to implement it; going through `cache(lsolver)` directly would tie those callers to one
method's cache layout.

Calling this before [`factorize!`](@ref) is an error, not `0` — an unfactorized solver is not
a non-singular one.
"""
function singular_index(lsolver::LinearSolver)
    error("No method `singular_index` implemented for method $(typeof(method(lsolver))).")
end

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
    error("No method for solve! implemented for this combination of input arguments $(typeof.(args)).")
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

Compare this to [`solve(::LinearSolver, ::AbstractVector)`](@ref).
"""
function solve!(::AbstractVector, ::LinearSolver, ::AbstractMatrix, ::AbstractVector)
    error("No method for solve! implemented for this combination of input arguments.")
end

"""
    solve(ls::LinearSolver, args...)

Counterpart of [`solve!`](@ref) for a prebuilt [`LinearSolver`](@ref): allocates
(and returns) a fresh solution vector instead of writing into a caller-supplied
one.  Note that the solver's *cache* is still updated in place (the
factorization is computed there).

Accepts the same trailing arguments as `solve!(ls, args...)`: a
[`LinearProblem`](@ref), a matrix-vector pair `A, b`, or a bare right-hand side
`b` (the latter uses the factorization already stored in `ls`).
"""
solve(ls::LinearSolver, args...) = solve!(ls, args...)

"""
    default_linear_solver_method(A)

The [`LinearSolverMethod`](@ref) to use for a system whose matrix looks like `A`.

This is what [`NewtonSolver`](@ref) and [`DogLegSolver`](@ref) fall back on when no
`linear_solver_method` is given, and it dispatches on the *Jacobian prototype* rather than on
the element type alone, because the right answer depends on the storage as much as on the
number type:

| `A` | method |
|---|---|
| dense, `Float32`/`Float64`/`ComplexF32`/`ComplexF64` | [`LapackLU`](@ref) |
| sparse, `Float64`/`ComplexF64` (or their 32-bit forms) | [`UmfpackLU`](@ref) |
| anything else — `BigFloat`, `Rational`, … | [`LU`](@ref) |

Note the last row covers a *sparse* matrix of a generic element type too, where [`LU`](@ref)
densifies it. [`SparspakLU`](@ref) is the method that would actually exploit the sparsity
there, and it is the only option in the package that can, but it lives in a package extension
and so may not be loaded — a default that errors depending on what the caller imported would
be worse than one that is merely slow. Pass `SparspakLU()` explicitly for a large sparse
`BigFloat` or `Rational` system.

[`LU`](@ref) used to be the default everywhere. It is the right choice for very small systems,
where its static-matrix cache allocates nothing, and the only choice for element types LAPACK
does not know — but its `MMatrix` path stops at [`N_STATIC_THRESHOLD`](@ref) `= 10`, and above
that it is a scalar triple loop with no blocking. Measured on an Apple M4 Max it is 2× slower
than [`LapackLU`](@ref) at `n = 64` and 32× slower at `n = 768`, with its triangular solve a
further 3.5–4.5× behind `getrs` throughout. A caller who did not know to pass
`linear_solver_method` paid all of that; downstream, that was 74 % of an implicit time step.

[`RecursiveLU`](@ref) is never selected automatically: it lives in a package extension, its
useful range depends on which BLAS is loaded, and it is not always installed. Choose it
explicitly.
"""
default_linear_solver_method(A::AbstractMatrix) =
    eltype(A) <: LinearAlgebra.BlasFloat ? LapackLU() : LU()

default_linear_solver_method(A::SparseMatrixCSC) =
    eltype(A) <: Union{Float64,ComplexF64,Float32,ComplexF32} ? UmfpackLU() : LU()
