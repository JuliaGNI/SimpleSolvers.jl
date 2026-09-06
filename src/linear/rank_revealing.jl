"""
    rank_tolerance(method, T)

The relative threshold below which a direction is treated as absent from the range of the
matrix, for a [`RankRevealingMethod`](@ref) applied to a matrix of element type `T`.

A method constructed with `rtol = missing` — the default — resolves to `sqrt(eps(real(T)))`
here, so that the same method object gives a tolerance appropriate to whichever precision it
is handed. An explicit `rtol` is converted to `real(T)` and used as given.

# The default, and why it is looser than `LinearAlgebra.rank`'s

`LinearAlgebra.rank` and `pinv` default to `min(size(A)...) * eps(real(T))` — for a `13 × 13`
`Float64` matrix about `2.9e-15`. That is the right threshold for *asking* what the rank is,
and the wrong one for *deciding what to invert*: it sits barely one order above the rounding
noise it is meant to separate, so a singular value that is exactly zero in exact arithmetic
lands on either side of it depending on the BLAS. Dividing by such a value is what turns a
harmless null direction into an enormous step.

`sqrt(eps(real(T)))` — `1.5e-8` for `Float64`, `3.5e-4` for `Float32` — leaves eight orders of
headroom on both sides. It is deliberately conservative:
a direction whose singular value is below it contributes less to the solution than it costs in
amplified error.

There is no default that is right for every problem. A matrix with a genuinely decaying
spectrum and no gap has no correct rank, and this number is then the whole answer — pass it
explicitly and say why.
"""
function rank_tolerance end

rank_tolerance(m::RankRevealingMethod, ::Type{T}) where {T} = convert(real(T), m.rtol)

function rank_tolerance(::RankRevealingMethod{Missing}, ::Type{T}) where {T}
    sqrt(eps(real(T)))
end

"""
    checkfactorized(lsolver::LinearSolver{T,<:RankRevealingMethod})

Throw an `ArgumentError` if [`factorize!`](@ref) has not been called on `lsolver` yet.

The [`RankRevealingMethod`](@ref) counterpart of the same guard for a
[`PivotedLUMethod`](@ref). It matters more here than there: an unfactorized cache has
`rank = 0`, which [`ldiv!`](@ref) would otherwise read as "the matrix is entirely null" and
answer with a zero vector rather than complain.
"""
function checkfactorized(lsolver::LinearSolver{
        T, LSM}) where {T, LSM <: RankRevealingMethod}
    cache(lsolver).factorized || throw(ArgumentError(
        "the $(nameof(LSM)) solver has not been factorized yet; call factorize! before ldiv!/solve!."))
    nothing
end

"""
    rank(lsolver::LinearSolver{T,<:RankRevealingMethod})

The numerical rank the factorization found, at the tolerance the method carries.

This extends `LinearAlgebra.rank`, and unlike it is read off a factorization that has already
been computed, so it costs nothing on top of the solve. It is the reason to reach for one of
these methods on a problem that is *not* failing: a system whose unknown count exceeds this
number carries that many degrees of freedom the residual cannot see.

# How far it agrees with `LinearAlgebra.rank`

For [`SVDSolver`](@ref) the two are the same computation — both count the singular values above
`rtol * σ₁` — so they agree whenever the `rtol` does.

For [`PivotedQR`](@ref) they usually agree and are not guaranteed to. The moduli of the `R`
diagonal only *bound* the singular values, so column pivoting can fail to expose a small one:
on a `70 × 70` Kahan matrix with ``\\theta = 1.15`` this returns `70`, where both
`LinearAlgebra.rank(A; rtol = sqrt(eps()))` and [`SVDSolver`](@ref) return `69`. The
disagreement is not confined to the reported number — the solve inherits it, and on that matrix
the `x` that comes back differs from the pseudoinverse solution by a relative `0.30`. Where the
rank itself is the answer rather than a means to one, use [`SVDSolver`](@ref).
"""
function LinearAlgebra.rank(lsolver::LinearSolver{
        T, LSM}) where {T, LSM <: RankRevealingMethod}
    checkfactorized(lsolver)
    cache(lsolver).rank
end

"""
    singular_index(lsolver::LinearSolver{T,<:RankRevealingMethod})

Always `0`, because these methods have no singular case: a rank-deficient matrix is solved
rather than reported.

The return value is not vacuous — it is what tells [`DogLegSolver`](@ref) that the Newton leg
of the step is available (see [`SimpleSolvers.directions!`](@ref)), which for a
[`PivotedLUMethod`](@ref) it would not be. Use [`rank`](@ref) to ask the question a zero pivot
index was standing in for.
"""
function singular_index(lsolver::LinearSolver{T, LSM}) where {T, LSM <: RankRevealingMethod}
    checkfactorized(lsolver)
    0
end

"""
    _rank_from_decreasing(d, rtol)

The number of leading entries of `d` that exceed `rtol * d[1]`, where `d` is a non-increasing
sequence of non-negative scalars — the singular values, or the moduli of the `R` diagonal of a
column-pivoted `QR`.

Written as a leading run rather than a `count`: a rank is a *prefix* of that sequence, and on
a matrix whose diagonal is not perfectly monotone (column pivoting guarantees it only up to
rounding) counting matches would produce a number that is not the size of any leading
submatrix, which is what the triangular solve then indexes.

`d` is any iterable, so a caller with the values already in a vector passes it and one reading
them off a diagonal passes a generator, without either of them materializing the other's form.
"""
function _rank_from_decreasing(d, rtol::Real)
    r = 0
    threshold = zero(rtol)
    for (i, dᵢ) in enumerate(d)
        i == 1 && (threshold = rtol * dᵢ)
        dᵢ > threshold || break
        r += 1
    end
    r
end

"""
    factorize!(lsolver::LinearSolver{T,<:RankRevealingMethod}[, A])

Decompose in place in `cache(lsolver)`, determining the numerical rank as it goes — with
whichever decomposition the method selects, see [`_decompose!`](@ref). With two arguments `A`
is first copied into the cache; with one, whatever the cache already holds is decomposed.

Unlike a [`PivotedLUMethod`](@ref) there is nothing here that [`ldiv!`](@ref) has to report
later: a deficient rank is an answer, not a failure. Read it with [`rank`](@ref).

!!! warning
    As for every other method here, the decomposition overwrites `cache(lsolver).A`, so the
    single-argument form is good for exactly one call. Use the two-argument form to
    refactorize.
"""
function factorize!(lsolver::LinearSolver{T, LSM}) where {T, LSM <: RankRevealingMethod}
    c = cache(lsolver)
    Base.require_one_based_indexing(c.A)
    _decompose!(method(lsolver), c)
    c.factorized = true
    lsolver
end

function factorize!(lsolver::LinearSolver{T, LSM}, A::AbstractMatrix{T}) where {
        T, LSM <: RankRevealingMethod}
    c = cache(lsolver)
    axes(A) == axes(c.A) || throw(DimensionMismatch(
        "the matrix to factorize has axes $(axes(A)), but the $(nameof(LSM)) cache was built " *
        "for $(axes(c.A)); allocate a new LinearSolver for a differently sized problem"))
    copyto!(c.A, A)
    factorize!(lsolver)
end

function factorize!(lsolver::LinearSolver{T, LSM}, ls::LinearProblem{T}) where {
        T, LSM <: RankRevealingMethod}
    factorize!(lsolver, matrix(ls))
end

"""
    _decompose!(method, cache)

Compute the decomposition of `cache.A` in place and record the numerical rank in
`cache.rank`.

This is the only thing that differs between the [`RankRevealingMethod`](@ref)s beyond the
[`ldiv!`](@ref) that consumes the result; the rank tolerance, [`factorize!`](@ref),
[`rank`](@ref), [`singular_index`](@ref) and every [`solve!`](@ref) form are shared.
"""
function _decompose! end

function solve!(solution::AbstractVector, lsolver::LinearSolver{T, LSM},
        ls::LinearProblem) where {T, LSM <: RankRevealingMethod}
    factorize!(lsolver, matrix(ls))
    ldiv!(solution, lsolver, rhs(ls))
    solution
end

function solve!(solution::AbstractVector, lsolver::LinearSolver{T, LSM},
        A::AbstractMatrix, b::AbstractVector) where {T, LSM <: RankRevealingMethod}
    factorize!(lsolver, A)
    ldiv!(solution, lsolver, b)
    solution
end

function solve!(solution::AbstractVector, lsolver::LinearSolver{T, LSM},
        b::AbstractVector) where {T, LSM <: RankRevealingMethod}
    ldiv!(solution, lsolver, b)
end

function solve!(lsolver::LinearSolver{T, LSM}, args...) where {
        T, LSM <: RankRevealingMethod}
    x = alloc_rhs(cache(lsolver).A)
    solve!(x, lsolver, args...)
    x
end

"""
    solve(method::RankRevealingMethod, ls::LinearProblem)
    solve(method::RankRevealingMethod, A, b)

Allocate a [`LinearSolver`](@ref), decompose and solve in one call.

The counterpart of [`solve(::PivotedLUMethod, ::LinearProblem)`](@ref). Convenient for a
one-off system; for a solve inside a loop, build the [`LinearSolver`](@ref) once and call
[`factorize!`](@ref) and [`ldiv!`](@ref) on it instead.
"""
function solve(method::RankRevealingMethod, ls::LinearProblem)
    lsolver = LinearSolver(method, ls)
    solve!(lsolver, ls)
end

function solve(method::RankRevealingMethod, A::AbstractMatrix, b::AbstractVector)
    solve(method, LinearProblem(A, b))
end

"""
    _blas_eltype_check(method, T)

Throw an `ArgumentError` naming `T` unless LAPACK provides it.

Both [`RankRevealingMethod`](@ref)s are LAPACK-backed — `geqp3`/`tzrzf` and `gesdd` — so both
are restricted to the same four element types as [`LapackLU`](@ref), and neither has a
self-contained fallback the way [`LU`](@ref) is one for `getrf`.
"""
function _blas_eltype_check(::LSM, ::Type{T}) where {LSM <: RankRevealingMethod, T}
    T <: LinearAlgebra.BlasFloat || throw(ArgumentError(
        "$(nameof(LSM)) is restricted to the element types LAPACK provides, i.e. Float32, " *
        "Float64, ComplexF32 and ComplexF64, but got $(T); there is no rank-revealing " *
        "method here for other element types"))
    nothing
end

# LAPACK spells the adjoint 'C' and the transpose 'T', and for a real matrix only 'T' is
# accepted — so the orthogonal factors of a complex decomposition have to be applied with the
# one and a real one with the other. This is shared by both `ldiv!`s.
_adjoint_char(::Type{<:Real}) = 'T'
_adjoint_char(::Type{<:Complex}) = 'C'
