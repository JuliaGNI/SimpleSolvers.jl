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
struct LinearSolver{T, LSMT <: LinearSolverMethod, LSCT <: LinearSolverCache} <:
       AbstractLinearSolver
    method::LSMT
    cache::LSCT

    function LinearSolver(method::LSMT,
            cache::LSCT) where {T, LSMT <: LinearSolverMethod, LSCT <: LinearSolverCache{T}}
        new{T, LSMT, LSCT}(method, cache)
    end
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

function LinearAlgebra.ldiv!(::AbstractVector, s::LinearSolver, ::AbstractVector)
    error("ldiv! not implemented for $(typeof(s))")
end

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
    resolve_linear_solver_method(linear_solver_method, A)

Resolve the [`LinearSolverMethod`](@ref) for a nonlinear-solver constructor: an explicit one
wins, `missing` falls back to [`default_linear_solver_method`](@ref) for the Jacobian prototype
`A`.

By dispatch rather than by `coalesce`, which would evaluate the fallback even when it is not
needed — and the fallback is allowed to *throw* (a sparse 32-bit float has no defensible
default), so an explicit method has to reach the solver without it being consulted at all.
"""
function resolve_linear_solver_method(linear_solver_method::LinearSolverMethod, ::AbstractMatrix)
    linear_solver_method
end

resolve_linear_solver_method(::Missing, A::AbstractMatrix) = default_linear_solver_method(A)

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
| dense, anything else — `BigFloat`, `Rational`, … | [`LU`](@ref) |
| sparse, `Float64`/`ComplexF64` | [`UmfpackLU`](@ref) |
| sparse, anything else | none — an `ArgumentError` |

**A sparse matrix is never densified for you.** [`UmfpackLU`](@ref), the sparse default, solves
`Float64` and `ComplexF64` systems only — UMFPACK converts a 32-bit matrix in `lu`/`lu!` but
has no 32-bit solve, and it does not handle `BigFloat` or `Rational` at all. For every other
element type this raises, naming the two things a caller might have meant:
[`SparspakLU`](@ref), which is generic in the element type and keeps the matrix sparse, or a
dense method that discards the sparsity.

Densifying is a real answer — it is often the right one for a small matrix — but it throws away
the structure the caller went to the trouble of building, and it is a decision that belongs to
them rather than to a fallback. Choosing [`SparspakLU`](@ref) instead is not available to a
default either: it lives in a package extension, so a default that reached for it would work or
fail depending on what the caller had imported. So this asks. Once asked, both answers work:
pass either as `linear_solver_method`.

[`LU`](@ref) is the choice for a dense matrix of an element type LAPACK does not know, and the
right one for a very small system, where its static-matrix cache allocates nothing. (A
`Rational` or `Integer` matrix reaches it and is then refused by [`lucache_eltype`](@ref),
which names the conversion to make — the package has no dense method for those at all, and
saying so is better than a default that pretends otherwise.)

But its allocation-free `MMatrix` path stops at [`N_STATIC_THRESHOLD`](@ref) `= 10` — and at any
size for a non-`isbitstype` element type, see [`_static`](@ref) — and above that it is a scalar
triple loop with no blocking. Measured on an Apple M4 Max it is 2× slower
than [`LapackLU`](@ref) at `n = 64` and 32× slower at `n = 768`, with its triangular solve a
further 3.5–4.5× behind `getrs` throughout. That is what the default spares a caller who does not
know to pass `linear_solver_method`: downstream it came to 74 % of an implicit time step.

[`RecursiveLU`](@ref) is never selected automatically: it lives in a package extension, its
useful range depends on which BLAS is loaded, and it is not always installed. Choose it
explicitly.
"""
function default_linear_solver_method(A::AbstractMatrix)
    eltype(A) <: LinearAlgebra.BlasFloat ? LapackLU() : LU()
end

# No densifying fallback: `LU` and `LapackLU` would both accept a sparse matrix here and
# quietly discard its structure, which is not a default's decision to make. See the docstring.
function default_linear_solver_method(A::SparseMatrixCSC)
    T = eltype(A)
    T <: Union{Float64, ComplexF64} && return UmfpackLU()
    throw(ArgumentError(
        "there is no default linear solver for a sparse matrix of element type $(T): the " *
        "sparse default, UmfpackLU, solves Float64 and ComplexF64 systems only, and a sparse " *
        "matrix is never densified for you. Pass linear_solver_method = SparspakLU() to keep " *
        "the matrix sparse — it is generic in the element type" *
        _sparse_dense_alternative(T) * "."))
end

# The densifying alternative to name, if there is one. `LU` accepts only floating-point element
# types (see `lucache_eltype`), so for a `Rational` or `Integer` matrix `SparspakLU` is the only
# method in the package that works at all and there is nothing to offer alongside it.
function _sparse_dense_alternative(::Type{T}) where {T}
    T <: LinearAlgebra.BlasFloat &&
        return " — or LapackLU() to densify it and solve it as a dense system"
    T <: Union{AbstractFloat, Complex{<:AbstractFloat}} &&
        return " — or LU() to densify it and solve it as a dense system"
    " — and, since LU accepts only floating-point matrices, the only method here that handles " *
    "this element type at all"
end
