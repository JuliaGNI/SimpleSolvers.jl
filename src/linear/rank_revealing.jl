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

# `Float16`

The eight orders above are a `Float64` statement. In `Float16` the default is `0.031`, and the
margins around it are one order rather than eight:

| | |
|---|---|
| the default, `sqrt(eps(Float16))` | `3.1e-2` |
| what rounding a deficient matrix into `Float16` already perturbs a zero singular value to | `~1e-3 · σ₁` |
| the orthogonality the [`SVDSolver`](@ref) sweeps reach | `~1e-4` |

So the default still sits a factor of thirty above the floor the *input* imposes, which is a
property of the data rather than of the algorithm and which no method can get under. It is
usable, and it is not comfortable. Where the rank itself is the answer at this precision, pass
an `rtol` and say why.

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

For the two singular value decompositions the two are the same computation — both count the
singular values above `rtol * σ₁` — so they agree whenever the `rtol` does.

For a pivoted `QR` they usually agree and are not guaranteed to. The moduli of the `R` diagonal
only *bound* the singular values, so column pivoting can fail to expose a small one. The
classical matrix on which it does is Kahan's, and the two pivoted `QR`s here do not fail on it
alike:

| `n` | ``\\theta`` | [`PivotedQR`](@ref) | [`LapackPivotedQR`](@ref) | either SVD, and `LinearAlgebra.rank` | margin |
|---:|---:|---:|---:|---:|---:|
| 70 | 1.15 | 69 | **70** | 69 | ``2\\cdot10^4`` |
| 90 | 1.15 | 89 | **90** | 89 | ``10^8`` |

The difference is the pivot search. [`LapackPivotedQR`](@ref) carries running column norms and
downdates them, which is asymptotically cheaper and drifts; [`PivotedQR`](@ref) recomputes them
exactly at every step and here keeps the pivot order that exposes the small direction. That is
a result on one adversarial family and not a guarantee — the bound is still only a bound, and
a matrix that defeats both exists. Where the rank itself is the answer rather than a means to
one, ask an SVD.

The last column is how far the smallest ``|R_{ii}|`` sits below the tolerance, and it is what
decides whether a row means anything. The effect is visible over a much wider band of `n` and
``\\theta`` than these two rows, but most of that band is a near miss: at ``n = 90``,
``\\theta = 1.35`` the margin is `1.2`, and a factor of `1.2` is a rounding difference between
two Julia versions rather than a property of the algorithm. Only a row that clears the
tolerance by orders is worth quoting, and only those two are pinned in the test suite.

The disagreement is not confined to the reported number: the solve inherits it, and on the
`70 × 70` matrix above the `x` [`LapackPivotedQR`](@ref) returns differs from the pseudoinverse
solution by a relative `0.30`.
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
    # cleared first, so that a `_decompose!` that throws — the Jacobi sweep cap is one way —
    # leaves the cache unusable rather than holding the previous matrix's rank and factors
    # behind a `factorized` flag that is still true.
    c.factorized = false
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

The gate of the two LAPACK-backed methods, [`LapackPivotedQR`](@ref) and
[`LapackSVDSolver`](@ref), which are restricted to the same four element types as
[`LapackLU`](@ref). Their pure-Julia counterparts [`PivotedQR`](@ref) and [`SVDSolver`](@ref)
take any floating-point type and use [`_float_eltype_check`](@ref) instead — the same split
[`LU`](@ref) and [`LapackLU`](@ref) have.
"""
function _blas_eltype_check(::LSM, ::Type{T}) where {LSM <: RankRevealingMethod, T}
    T <: LinearAlgebra.BlasFloat || throw(ArgumentError(
        "$(nameof(LSM)) is restricted to the element types LAPACK provides, i.e. Float32, " *
        "Float64, ComplexF32 and ComplexF64, but got $(T); drop the `Lapack` prefix for a " *
        "method that factorizes any floating-point type in plain Julia"))
    nothing
end

"""
    _float_eltype_check(method, T)

Throw an `ArgumentError` naming `T` unless it is a floating-point type.

The gate of the two pure-Julia methods, [`PivotedQR`](@ref) and [`SVDSolver`](@ref). It is the
same contract [`lucache_eltype`](@ref) enforces for [`LU`](@ref), and for the same reason: a
Householder reflector and a Jacobi rotation are both built out of square roots, which a
`Rational` or an `Integer` does not have.
"""
function _float_eltype_check(::LSM, ::Type{T}) where {LSM <: RankRevealingMethod, T}
    T <: AbstractFloat || T <: Complex{<:AbstractFloat} ||
        throw(ArgumentError(
            "$(nameof(LSM)) only supports floating-point element types (AbstractFloat or " *
            "Complex{<:AbstractFloat}); got $(T). Convert the problem to a floating-point type " *
            "first, e.g. `float.(A)`."))
    nothing
end

"""
    _prescale!(A) -> s

Scale `A` in place by a power of two `s` chosen so that `maximum(abs, A) ≤ 1`, and return `s`.

A power of two moves an exponent and leaves every mantissa alone, so scaling up is exact and
scaling down is exact for every entry that stays normal. It exists for `Float16`, which
overflows at `65504`: a column of entries in the hundreds has a sum of squares that does not
fit, where the same column scaled below one cannot overflow for any `n` a linear solver will
see. Every norm, dot product and reflector application in the two pure-Julia decompositions is
safe because of this one pass, which is why none of them carries a scaling of its own.

The exception is scaling down far enough to drive entries subnormal, which in `Float16` starts
at a spread of about `2^11`: those entries do lose mantissa bits, and below `6.1e-5` relative
they flush to zero. Both methods read a rank off a threshold relative to the largest `R`
diagonal or singular value, and that threshold sits far above the subnormal range, so what is
lost is already below the resolution either method reports.

Both callers undo it by scaling the solution rather than the factors, since the `x` that
solves `sAx = b` is `1/s` times the one that solves `Ax = b`. [`PivotedQR`](@ref) multiplies
`c.y` on its way into `x`; [`SVDSolver`](@ref) divides by `S * s` and multiplies `x` at the
end, which keeps the quotient from overflowing for a matrix small enough that `s` is large.
`S` itself is stored unscaled, so [`singular_values`](@ref) needs no correction.
"""
function _prescale!(A::AbstractMatrix{T}) where {T}
    RT = real(T)
    m = maximum(abs, A; init = zero(RT))
    (iszero(m) || !isfinite(m)) && return one(RT)
    # A subnormal `m` — in `Float16` that is every entry below `6.1e-5` — asks for a factor
    # larger than `RT` can hold, and an infinite `s` would turn the whole matrix into `Inf`.
    # Capping the exponent keeps `s` finite, and `maximum(abs, A) ≤ 1` still holds because a
    # capped `s` is `2^E` against an `m` below `2^-E`.
    s = ldexp(one(RT), min(-exponent(m) - 1, exponent(floatmax(RT))))
    isone(s) || (A .*= s)
    s
end

"""
    _accumulator(T)

The type an inner product over a column is summed in: `Float32` for `Float16`, and `T` itself
for everything else.

`Float16` arithmetic rounds to eleven bits after *every* operation, so a dot product of `n`
terms loses far more than the `eps(Float16)` a single one does. Measured on a random
`13 × 13` matrix, the one-sided Jacobi sweep of [`SVDSolver`](@ref) cannot drive the relative
off-diagonality below `3.1e-2` when the three inner products of a rotation are summed in
`Float16` — which is `32 · eps(Float16)`, and coarser than
`rank_tolerance(SVDSolver(), Float16)` itself. The singular values would then carry an error
of the same size as the threshold deciding which of them are zero, and the method could not
answer the question it exists for. Summing in `Float32` puts the floor back at
`eps(Float16)`.

Storage, results and the reported singular values stay in `T`. What widens is the running sum
inside [`_acc_dot`](@ref) and [`_acc_norm`](@ref), and with it the rotation and reflector
coefficients computed from them — one rounding per stored entry instead of `n`.
"""
_accumulator(::Type{Float16}) = Float32
_accumulator(::Type{ComplexF16}) = ComplexF32
_accumulator(::Type{T}) where {T} = T

"""
    _acc_dot(x, y)

``x^* y``, summed in [`_accumulator`](@ref)`(eltype(x))` and returned in it.

No scaling, unlike `LinearAlgebra.dot`: [`_prescale!`](@ref) has already brought every entry
below one in modulus, so a sum of `n` products cannot overflow and only entries far below the
level that could affect the result underflow.
"""
function _acc_dot(x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    AT = _accumulator(T)
    s = zero(AT)
    @inbounds for i in eachindex(x, y)
        s += conj(AT(x[i])) * AT(y[i])
    end
    s
end

"""
    _acc_norm(x)

``\\lVert x \\rVert_2``, summed as [`_acc_dot`](@ref) does and returned in its real type.
"""
_acc_norm(x::AbstractVector) = sqrt(real(_acc_dot(x, x)))

"""
    singular_values(lsolver)

The singular values of the matrix that was factorized, in non-increasing order.

Defined for [`SVDSolver`](@ref) and [`LapackSVDSolver`](@ref). The array is the cache's own, so
it is overwritten by the next [`factorize!`](@ref) — copy it if it has to outlive one.

There is no counterpart for either pivoted `QR`: the moduli of an `R` diagonal bound the
singular values but are not equal to them, and reporting them under this name would invite
reading a spectrum off numbers that are not one.
"""
function singular_values end

# LAPACK spells the adjoint 'C' and the transpose 'T', and for a real matrix only 'T' is
# accepted — so the orthogonal factors of a complex decomposition have to be applied with the
# one and a real one with the other. This is shared by both `ldiv!`s.
_adjoint_char(::Type{<:Real}) = 'T'
_adjoint_char(::Type{<:Complex}) = 'C'
