"""
    SparseFactorizationCache <: LinearSolverCache

The cache shared by the [`SparseDirectMethod`](@ref)s, [`UmfpackLU`](@ref) and
[`SparspakLU`](@ref).

# Keys
- `F`: the backend's factorization object, which owns the ordering, the symbolic
  factorization and the numeric factors,
- `n`: the leading dimension, kept here because the two backends store it differently,
- `info`: `0` if the last factorization or solve found nothing wrong, non-zero otherwise; see
  [`singular_index`](@ref),
- `factorized`: whether [`factorize!`](@ref) has run at all.

Unlike [`PivotedLUCache`](@ref) there is no working copy of the matrix: both backends take the
`SparseMatrixCSC` as an argument to their refactorize call and read its `nzval` directly, so a
copy here would be dead weight. The consequence is that the *pattern* is fixed at
construction — which is exactly the contract a Newton loop wants, since reusing the ordering
and symbolic factorization is where the saving is.

Neither backend is allocation-free, and that is inherent to them rather than to this wrapper.
Measured on a periodic banded matrix at `n = 384`: [`UmfpackLU`](@ref) allocates ~374 kB per
refactorization but **0 B** per [`ldiv!`](@ref); [`SparspakLU`](@ref) allocates ~11 kB per
refactorization and ~10 kB per [`ldiv!`](@ref).
"""
mutable struct SparseFactorizationCache{T,FT} <: LinearSolverCache{T}
    F::FT
    n::Int
    info::Int
    factorized::Bool
end

SparseFactorizationCache{T}(F::FT, n::Integer) where {T,FT} =
    SparseFactorizationCache{T,FT}(F, Int(n), 0, false)

"""
    checkfactorized(lsolver::LinearSolver{T,<:SparseDirectMethod})

Throw an `ArgumentError` if [`factorize!`](@ref) has not been called on `lsolver` yet.

The [`SparseDirectMethod`](@ref) counterpart of the [`PivotedLUMethod`](@ref) guard.
"""
function checkfactorized(lsolver::LinearSolver{T,LSM}) where {T,LSM<:SparseDirectMethod}
    cache(lsolver).factorized || throw(ArgumentError(
        "the $(nameof(LSM)) solver has not been factorized yet; call factorize! before ldiv!/solve!."))
    nothing
end

"""
    singular_index(lsolver::LinearSolver{T,<:SparseDirectMethod})

`0` if the factorization succeeded, non-zero if it did not.

!!! warning "Not an index"
    Neither sparse backend reports *which* pivot vanished, so unlike
    [`LU`](@ref)/[`LapackLU`](@ref) this is a flag widened to the interface's return type, not
    a position. For [`SparspakLU`](@ref) it is worse than that: singularity is only detected
    when the factorization is *used*, so this returns `0` until a [`ldiv!`](@ref) has failed.
    See the [`SparspakLU`](@ref) docstring.
"""
function singular_index(lsolver::LinearSolver{T,LSM}) where {T,LSM<:SparseDirectMethod}
    checkfactorized(lsolver)
    cache(lsolver).info
end

"""
    checksparse(method, A)

Throw an `ArgumentError` unless `A` is a square `SparseMatrixCSC`.

The [`SparseDirectMethod`](@ref)s need the sparsity pattern, and handing one a dense matrix is
a mistake worth naming rather than silently converting: converting would build a
`SparseMatrixCSC` with no structural zeros, whose factorization is slower than
[`LapackLU`](@ref)'s in every measurement.
"""
function checksparse(method::SparseDirectMethod, A::AbstractMatrix)
    A isa SparseMatrixCSC || throw(ArgumentError(
        "$(nameof(typeof(method))) needs a SparseMatrixCSC — it factorizes the sparsity " *
        "pattern — but got $(typeof(A)); use LapackLU() for a dense matrix"))
    checksquare(A)
end

"""
    checkpattern(lsolver, A)

Throw a `DimensionMismatch` unless `A` matches the size the cache was built for.

The pattern itself is checked by the backend, which is where the useful error lives: both are
told the pattern may not change, because the ordering and symbolic factorization the cache
holds were computed for one.
"""
function checkpattern(lsolver::LinearSolver{T,LSM}, A::AbstractMatrix) where {T,LSM<:SparseDirectMethod}
    n = checksparse(method(lsolver), A)
    n == cache(lsolver).n || throw(DimensionMismatch(
        "the matrix to factorize is $(n)×$(n), but the $(nameof(LSM)) cache was built for " *
        "$(cache(lsolver).n)×$(cache(lsolver).n); allocate a new LinearSolver for a " *
        "differently sized problem"))
    nothing
end

factorize!(lsolver::LinearSolver{T,LSM}, ls::LinearProblem{T}) where {T,LSM<:SparseDirectMethod} =
    factorize!(lsolver, matrix(ls))

# The one-argument form has nothing to factorize: unlike `PivotedLUCache`, this cache holds no
# copy of the matrix (see its docstring), so there is no "whatever the cache already holds".
function factorize!(lsolver::LinearSolver{T,LSM}) where {T,LSM<:SparseDirectMethod}
    error("$(nameof(LSM)) has no single-argument factorize!: the cache holds a factorization, " *
          "not a copy of the matrix. Call factorize!(lsolver, A).")
end

function LinearAlgebra.ldiv!(x::AbstractVector{T}, lsolver::LinearSolver{T,LSM}, b::AbstractVector{T}) where {T,LSM<:SparseDirectMethod}
    c = cache(lsolver)
    checkfactorized(lsolver)
    @assert axes(x, 1) == axes(b, 1) == Base.OneTo(c.n)
    Base.require_one_based_indexing(x, b)
    c.info == 0 || throw(SingularException(c.info))
    _sparse_ldiv!(method(lsolver), c, x, b)
    x
end

"""
    _sparse_ldiv!(method, cache, x, b)

Solve with the factorization in `cache`, writing the result to `x`, and report a singular
factorization as a `SingularException`.

The per-method half of [`ldiv!`](@ref) for a [`SparseDirectMethod`](@ref); the shared half
does the guards. The two backends differ in *when* they can tell that the matrix was singular,
which is why this is not shared. See [`singular_index`](@ref).
"""
function _sparse_ldiv! end

function solve!(solution::AbstractVector, lsolver::LinearSolver{T,LSM}, ls::LinearProblem) where {T,LSM<:SparseDirectMethod}
    factorize!(lsolver, matrix(ls))
    ldiv!(solution, lsolver, rhs(ls))
    solution
end

function solve!(solution::AbstractVector, lsolver::LinearSolver{T,LSM}, A::AbstractMatrix, b::AbstractVector) where {T,LSM<:SparseDirectMethod}
    factorize!(lsolver, A)
    ldiv!(solution, lsolver, b)
    solution
end

function solve!(solution::AbstractVector, lsolver::LinearSolver{T,LSM}, b::AbstractVector) where {T,LSM<:SparseDirectMethod}
    ldiv!(solution, lsolver, b)
end

function solve!(lsolver::LinearSolver{T,LSM}, args...) where {T,LSM<:SparseDirectMethod}
    x = fill(_nan(T), cache(lsolver).n)
    solve!(x, lsolver, args...)
    x
end

"""
    solve(method::SparseDirectMethod, ls::LinearProblem)
    solve(method::SparseDirectMethod, A, b)

Allocate a [`LinearSolver`](@ref), factorize and solve in one call.

For a one-off system only: this pays the ordering and symbolic factorization every time, which
is most of the cost of a sparse solve. Inside a loop, build the [`LinearSolver`](@ref) once
and call [`factorize!`](@ref) and [`ldiv!`](@ref) on it — that is what reuses the symbolic
phase.
"""
function solve(method::SparseDirectMethod, ls::LinearProblem)
    lsolver = LinearSolver(method, ls)
    solve!(lsolver, ls)
end

solve(method::SparseDirectMethod, A::AbstractMatrix, b::AbstractVector) =
    solve(method, LinearProblem(A, b))
