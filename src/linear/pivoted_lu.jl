"""
    PivotedLUCache <: LinearSolverCache

The cache shared by every [`PivotedLUMethod`](@ref) — [`LapackLU`](@ref) and
[`RecursiveLU`](@ref).

# Keys
- `A`: the working copy of the matrix, which the factorization overwrites in place,
- `ipiv`: the pivot vector the factorization fills,
- `info`: the index of the first zero pivot, or `0` on success,
- `factorized`: whether [`factorize!`](@ref) has run at all, which `info == 0` cannot express.

The cache holds the *pieces* of the factorization rather than a `LinearAlgebra.LU` object, so
that [`factorize!`](@ref) can be called any number of times without allocating: `LU` is
immutable, so a fresh one would have to be built and boxed into a field on every
refactorization, and its `ipiv` freshly allocated. Both are `O(n)`-to-`O(1)` costs against an
`O(n^3)` factorization, but a nonlinear solve in a time-stepping loop refactorizes on every
step of every step, and [`LU`](@ref) — the method these sit beside — allocates nothing at all.

Use [`factorization`](@ref) to get a `LinearAlgebra.LU` view of these pieces when one is
actually wanted (for `det`, say).

The pivot vector is typed `Vector{LinearAlgebra.BlasInt}` rather than `Vector{Int}`: that is
what `getrf` fills, and the two are not the same type under a 32-bit-integer BLAS. It is also
what [`RecursiveLU`](@ref) writes into, whose `ipiv` is any `AbstractVector{<:Integer}` — so
one cache type serves both, and the LAPACK triangular solve can be shared with it.
"""
mutable struct PivotedLUCache{T, AT <: AbstractMatrix{T}} <: LinearSolverCache{T}
    A::AT
    ipiv::Vector{LinearAlgebra.BlasInt}
    info::LinearAlgebra.BlasInt
    factorized::Bool
end

"""
    _pivoted_lu_cache(A)

Build a [`PivotedLUCache`](@ref) from `A`.

Any `AbstractMatrix` storage is accepted, but the cache always holds a plain `Matrix`, because
that is what both backends can be pointed at. It is a copy rather than `undef` so that the
single-argument [`factorize!`](@ref) has something to factorize — as it does for [`LU`](@ref),
whose cache is likewise seeded from `A`.
"""
function _pivoted_lu_cache(A::AbstractMatrix{T}) where {T}
    n = checksquare(A)
    Ā = Matrix{T}(A)
    PivotedLUCache{T, typeof(Ā)}(Ā, zeros(LinearAlgebra.BlasInt, n), 0, false)
end

"""
    checkfactorized(lsolver::LinearSolver{T,<:PivotedLUMethod})

Throw an `ArgumentError` if [`factorize!`](@ref) has not been called on `lsolver` yet.

The [`PivotedLUMethod`](@ref) counterpart of the `perms[1] == 0` guard in [`LU`](@ref)'s
[`ldiv!`](@ref): without it, `getrs` would be handed an all-zero pivot vector and return
garbage rather than complain.
"""
function checkfactorized(lsolver::LinearSolver{T, LSM}) where {T, LSM <: PivotedLUMethod}
    cache(lsolver).factorized || throw(ArgumentError(
        "the $(nameof(LSM)) solver has not been factorized yet; call factorize! before ldiv!/solve!."))
    nothing
end

"""
    factorization(lsolver::LinearSolver{T,<:PivotedLUMethod})

A `LinearAlgebra.LU` view of the factorization held in the cache.

This wraps the cache's arrays rather than copying them, so it is only valid until the next
[`factorize!`](@ref). Neither [`ldiv!`](@ref) nor [`singular_index`](@ref) goes through it —
building it allocates, and they are on the hot path — but it is what to reach for when a
`LinearAlgebra.Factorization` is what you want, e.g. `det(factorization(lsolver))`.
"""
function factorization(lsolver::LinearSolver{T, LSM}) where {T, LSM <: PivotedLUMethod}
    checkfactorized(lsolver)
    c = cache(lsolver)
    LinearAlgebra.LU(c.A, c.ipiv, c.info)
end

"""
    singular_index(lsolver::LinearSolver{T,<:PivotedLUMethod})

The zero-pivot index the factorization reported (LAPACK's `info`), or `0` if it succeeded.
"""
function singular_index(lsolver::LinearSolver{T, LSM}) where {T, LSM <: PivotedLUMethod}
    checkfactorized(lsolver)
    Int(cache(lsolver).info)
end

"""
    factorize!(lsolver::LinearSolver{T,<:PivotedLUMethod}[, A])

Factorize in place in `cache(lsolver).A`, with whichever kernel the method selects — see
[`_getrf!`](@ref). With two arguments `A` is first copied into the cache; with one, whatever
the cache already holds is factorized.

The factorization is not checked here. A singular matrix is reported when the factorization
is *used*, by [`ldiv!`](@ref), so that a caller that factorizes speculatively — as a
quasi-Newton method does — is not interrupted by a matrix it may never solve with.

!!! warning
    As for [`LU`](@ref), the factorization overwrites `cache(lsolver).A` with the factors, so
    the single-argument form is good for exactly one call; calling it twice would factorize
    the factors. Use the two-argument form to refactorize.
"""
function factorize!(lsolver::LinearSolver{T, LSM}) where {T, LSM <: PivotedLUMethod}
    c = cache(lsolver)
    Base.require_one_based_indexing(c.A)
    _getrf!(method(lsolver), c)
    c.factorized = true
    lsolver
end

function factorize!(lsolver::LinearSolver{T, LSM}, A::AbstractMatrix{T}) where {
        T, LSM <: PivotedLUMethod}
    c = cache(lsolver)
    axes(A) == axes(c.A) || throw(DimensionMismatch(
        "the matrix to factorize has axes $(axes(A)), but the $(nameof(LSM)) cache was built " *
        "for $(axes(c.A)); allocate a new LinearSolver for a differently sized problem"))
    copyto!(c.A, A)
    factorize!(lsolver)
end

function factorize!(lsolver::LinearSolver{T, LSM}, ls::LinearProblem{T}) where {
        T, LSM <: PivotedLUMethod}
    factorize!(lsolver, matrix(ls))
end

function LinearAlgebra.ldiv!(x::AbstractVector{T}, lsolver::LinearSolver{T, LSM},
        b::AbstractVector{T}) where {T, LSM <: PivotedLUMethod}
    c = cache(lsolver)
    checkfactorized(lsolver)
    @assert axes(x, 1) == axes(b, 1) == axes(c.A, 1)
    Base.require_one_based_indexing(x, b)
    c.info == 0 || throw(SingularException(Int(c.info)))
    # the solve is in place, so the right-hand side has to be moved into `x` first; this is a
    # no-op when the caller passes the same vector for both
    x === b || copyto!(x, b)
    _getrs!(c, x)
    x
end

# `getrs` is handed a pointer and a length, so it needs `x` contiguous — which every caller
# in the package and every `Vector` is, and which is why `ldiv!` allocates nothing.
#
# `LU` accepts any one-based `AbstractVector`, because it solves with scalar loops, so
# anything else goes through a contiguous scratch vector rather than becoming an error the
# other method does not have. (`LinearAlgebra.ldiv!(::LU{<:Any,<:StridedMatrix}, x)` is no
# help here: it reaches the same `getrs` and the same "matrix does not have contiguous
# columns" for, say, a stride-2 view.)
#
# This is shared with `RecursiveLU`: RecursiveFactorization writes LAPACK-layout factors with
# LAPACK's pivot convention, so the same triangular solve applies to both, and it is the
# faster of the two solves available (against `LU`'s scalar loops) for either.
function _getrs!(c::PivotedLUCache{T}, x::AbstractVector{T}) where {T}
    if x isa StridedVector{T} && stride(x, 1) == 1
        LinearAlgebra.LAPACK.getrs!('N', c.A, c.ipiv, x)
    else
        y = Vector{T}(x)
        LinearAlgebra.LAPACK.getrs!('N', c.A, c.ipiv, y)
        copyto!(x, y)
    end
    x
end

function solve!(solution::AbstractVector, lsolver::LinearSolver{T, LSM},
        ls::LinearProblem) where {T, LSM <: PivotedLUMethod}
    factorize!(lsolver, matrix(ls))
    ldiv!(solution, lsolver, rhs(ls))
    solution
end

function solve!(solution::AbstractVector, lsolver::LinearSolver{T, LSM},
        A::AbstractMatrix, b::AbstractVector) where {T, LSM <: PivotedLUMethod}
    factorize!(lsolver, A)
    ldiv!(solution, lsolver, b)
    solution
end

function solve!(solution::AbstractVector, lsolver::LinearSolver{T, LSM},
        b::AbstractVector) where {T, LSM <: PivotedLUMethod}
    ldiv!(solution, lsolver, b)
end

function solve!(lsolver::LinearSolver{T, LSM}, args...) where {T, LSM <: PivotedLUMethod}
    x = alloc_rhs(cache(lsolver).A)
    solve!(x, lsolver, args...)
    x
end

"""
    solve(method::PivotedLUMethod, ls::LinearProblem)
    solve(method::PivotedLUMethod, A, b)

Allocate a [`LinearSolver`](@ref), factorize and solve in one call.

The counterpart of [`solve(::LU, ::LinearProblem)`](@ref). Convenient for a one-off system;
for a solve inside a loop, build the [`LinearSolver`](@ref) once and call
[`factorize!`](@ref) and [`ldiv!`](@ref) on it instead.
"""
function solve(method::PivotedLUMethod, ls::LinearProblem)
    lsolver = LinearSolver(method, ls)
    solve!(lsolver, ls)
end

function solve(method::PivotedLUMethod, A::AbstractMatrix, b::AbstractVector)
    solve(method, LinearProblem(A, b))
end
