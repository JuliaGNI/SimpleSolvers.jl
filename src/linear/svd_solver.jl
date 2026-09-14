"""
    struct SVDSolver{RT} <: RankRevealingMethod{RT}

A singular value decomposition written in plain Julia, meant to solve a
[`LinearProblem`](@ref) whose matrix is rank deficient — and to say how deficient it is.

Like [`PivotedQR`](@ref) it returns the **minimum-norm solution** rather than raising a
`SingularException`; unlike it, the decomposition it computes is the one the question is
usually asked in. [`singular_values`](@ref) hands back the spectrum the rank was read off, so
"is this matrix rank deficient, or merely ill-conditioned?" is answered by the same call that
solved the system — a gap of several orders in that spectrum is the first, a smooth decay the
second, and no rank tolerance can tell them apart on its own.

[`LapackSVDSolver`](@ref) computes the same thing through `gesdd` and is the faster of the two
on the four element types LAPACK has. This one is the counterpart that takes **any
floating-point element type**, `Float16` and `BigFloat` included — the same relationship
[`LU`](@ref) has to [`LapackLU`](@ref).

# Constructor

```jldoctest; setup = :(using SimpleSolvers)
SVDSolver()

# output

SVDSolver{Missing}(missing)
```

`rtol` is the relative threshold below which a singular value is treated as zero; `missing`
resolves it from the element type. See [`rank_tolerance`](@ref).

```jldoctest; setup = :(using SimpleSolvers)
SVDSolver(; rtol = 1e-10)

# output

SVDSolver{Float64}(1.0e-10)
```

# What it computes

A **one-sided Jacobi** iteration: plane rotations applied to pairs of *columns* until they are
mutually orthogonal, at which point the column norms are the singular values and the columns
themselves are ``\\sigma_j u_j``. The numerical rank is the number of leading singular values
above the tolerance, and [`ldiv!`](@ref) applies the truncated pseudo-inverse
``V \\Sigma_r^{-1} U^*``, which is the minimum-norm least-squares solution by construction.

Jacobi rather than the bidiagonalization-and-implicit-shift-`QR` that `gesdd` performs, for
three reasons that all matter more here than speed does. It **terminates unconditionally** —
the off-diagonal mass decreases at every single rotation, so there is no shift strategy and no
deflation threshold to be tuned per precision, which for `Float16` there is no body of
experience to tune against. It delivers **relative** accuracy on each singular value rather
than accuracy relative to ``\\sigma_1``, and a relative threshold on small singular values is
precisely what this method is asked for. And it needs no assembly pass: the sweeps turn the
working matrix into `U` in place, so the whole factorization runs out of a cache allocated
once.

# Cost

The decomposition costs a few times [`PivotedQR`](@ref)'s — six to twelve sweeps of
``O(n^3)`` each — where the *solve* is the cheap one, two matrix-vector products. So the sum
favours [`PivotedQR`](@ref) at one solve per factorization, which is what a Newton step is, and
this method when a factorization is reused across several right-hand sides or when the spectrum
is part of what is wanted.

**Both [`factorize!`](@ref) and [`ldiv!`](@ref) are allocation-free** for any `isbitstype`
element type, which neither LAPACK-backed method is. `BigFloat` allocates per arithmetic
operation and nothing here can change that.

No rank-revealing method is the default for anything — see
[`default_linear_solver_method`](@ref). Square matrices only, exactly as [`PivotedQR`](@ref).

# Example

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: singular_values; using LinearAlgebra: rank, ldiv!)
julia> A = Float16[1 2; 2 4];            # rank 1

julia> ls = LinearSolver(SVDSolver(), A);

julia> factorize!(ls, A);

julia> rank(ls)
1

julia> round.(Float64.(singular_values(ls)); digits = 2)
2-element Vector{Float64}:
 5.0
 0.0

julia> round.(Float64.(ldiv!(zeros(Float16, 2), ls, Float16[1, 2])); digits = 2)
2-element Vector{Float64}:
 0.2
 0.4
```
"""
struct SVDSolver{RT} <: RankRevealingMethod{RT}
    rtol::RT

    SVDSolver(; rtol = missing) = new{typeof(rtol)}(rtol)
end

"""
    SVDCache <: LinearSolverCache

The cache of an [`SVDSolver`](@ref).

# Keys
- `A`: the working copy of the matrix, which the sweeps turn into `U`,
- `V`: the right factor, held as columns rather than as ``V^*``,
- `S`: the singular values, non-increasing, and real even for a complex matrix,
- `y`: an `n`-vector holding ``U^* b`` while it is scaled, so that [`ldiv!`](@ref) allocates
  nothing and tolerates `x === b`,
- `scale`: the power of two [`_prescale!`](@ref) multiplied the matrix by,
- `rank`: the numerical rank [`factorize!`](@ref) found,
- `factorized`: whether [`factorize!`](@ref) has run at all, which `rank == 0` cannot express.

`S` holds the singular values of the matrix the caller passed, not of the scaled one, so
[`singular_values`](@ref) needs no correction. [`ldiv!`](@ref) divides by `S * scale` instead
and applies `scale` to the solution, which keeps the quotient in the range the scaling was
chosen to make safe — see [`_prescale!`](@ref).

There is no `U` field, and that is the point of the algorithm rather than an omission: a
one-sided Jacobi sweep rotates the columns of `A` in place, leaving ``\\sigma_j u_j`` in column
`j`, so normalizing the columns turns `A` itself into `U`. `A` is documented as destroyed by
[`factorize!`](@ref) anyway. `V` is stored by columns because one rotation kernel then serves
both arrays.

Nothing is reassigned — every array is allocated at construction and written into thereafter,
which is what makes [`factorize!`](@ref) allocation-free where
[`LapackSVDCache`](@ref)'s cannot be.
"""
mutable struct SVDCache{T, RT <: Real, AT <: AbstractMatrix{T}} <: LinearSolverCache{T}
    A::AT
    V::Matrix{T}
    S::Vector{RT}
    y::Vector{T}
    scale::RT
    rank::Int
    factorized::Bool
end

function LinearSolverCache(method::SVDSolver, A::AbstractMatrix{T}) where {T}
    _float_eltype_check(method, T)
    n = checksquare(A)
    Ā = Matrix{T}(A)
    SVDCache{T, real(T), typeof(Ā)}(Ā, zeros(T, n, n), zeros(real(T), n), zeros(T, n),
        one(real(T)), 0, false)
end

"""
    JACOBI_MAX_SWEEPS

The number of one-sided Jacobi sweeps [`_decompose!`](@ref) will run before giving up, the same
cap LAPACK's `gesvj` uses. Six to twelve is typical, and reaching thirty means something is
wrong rather than merely slow.
"""
const JACOBI_MAX_SWEEPS = 30

"""
    _rotate_columns!(A, p, q, cs, sn)

Apply the plane rotation `[cs -conj(sn); sn cs]` to columns `p` and `q` of `A`, in place.
"""
function _rotate_columns!(A::AbstractMatrix, p::Integer, q::Integer, cs, sn)
    @inbounds for i in axes(A, 1)
        u = A[i, p]
        v = A[i, q]
        A[i, p] = cs * u - conj(sn) * v
        A[i, q] = sn * u + cs * v
    end
    A
end

"""
    _jacobi_sweep!(A, V, tol) -> Int

One cyclic sweep of the one-sided Jacobi iteration over every pair of columns of `A`, applying
each rotation to `V` as well, and returning how many rotations were applied.

For a pair with ``\\alpha = \\lVert a_p\\rVert^2``, ``\\beta = \\lVert a_q\\rVert^2`` and
``\\gamma = a_p^* a_q``, the rotation that makes the two columns orthogonal has
``t = \\tan\\theta`` solving ``t^2 + 2\\zeta t - 1 = 0`` with
``\\zeta = (\\beta - \\alpha)/2\\lvert\\gamma\\rvert``. Taking the root of smaller modulus
keeps ``\\lvert\\theta\\rvert \\le \\pi/4``, which is what makes the sweep converge rather than
merely rearrange. For a complex matrix the sine additionally carries the phase of ``\\gamma``;
for a real one that factor is ``\\pm 1`` and the formula is the classical one.

A pair already orthogonal to within `tol` is skipped, and a sweep that skips every pair is the
fixed point.

# Columns that are numerically zero

A pair is also skipped when either column's norm falls below `tol` times the largest in the
matrix. Without that guard the sweep never terminates on a rank-deficient matrix: the
orthogonality test is relative to the pair itself, so two columns that hold nothing but
rounding noise are asked to become orthogonal to each other, and each rotation regenerates as
much noise as it removes. In `Float16` those columns are often *subnormal* — a rank-5
`13 × 13` matrix rounded to `Float16` leaves null-space columns of norm `5e-6`, where the
smallest normal number is `6e-5` — so they cannot be rotated accurately at all.

Nothing is lost by skipping them. A column below `tol` relative to the largest carries a
singular value below `√n · eps`, orders under any usable [`rank_tolerance`](@ref), so it is
truncated by the rank test either way; and its component along a *large* column perturbs that
column's singular value only in the second order.
"""
function _jacobi_sweep!(A::AbstractMatrix{T}, V::AbstractMatrix{T}, tol::Real) where {T}
    n = size(A, 2)
    rotations = 0

    αmax = zero(real(_accumulator(T)))
    for j in 1:n
        αmax = max(αmax, real(_acc_dot(view(A, :, j), view(A, :, j))))
    end
    # `αmax` first, so the whole product stays in the accumulator's type: `tol * tol` alone is
    # subnormal in `Float16` for every `n` a linear solver will see.
    negligible = αmax * tol * tol

    for p in 1:(n - 1), q in (p + 1):n

        aₚ = view(A, :, p)
        a_q = view(A, :, q)
        α = real(_acc_dot(aₚ, aₚ))
        β = real(_acc_dot(a_q, a_q))
        min(α, β) ≤ negligible && continue
        γ = _acc_dot(aₚ, a_q)
        abs(γ) ≤ tol * sqrt(α * β) && continue
        rotations += 1

        ζ = (β - α) / (2 * abs(γ))
        # `copysign` and not `sign`: the two agree except at ζ = 0, where the columns have
        # equal norms and the correct rotation is the full π/4 that `sign` would report as
        # none at all.
        t = copysign(one(ζ), ζ) / (abs(ζ) + sqrt(one(ζ) + ζ * ζ))
        cs = inv(sqrt(one(t) + t * t))
        sn = (γ / abs(γ)) * (cs * t)

        _rotate_columns!(A, p, q, cs, sn)
        _rotate_columns!(V, p, q, cs, sn)
    end
    rotations
end

"""
    _sort_singular_values!(c::SVDCache)

Order `c.S` non-increasingly, carrying the matching columns of `c.A` and `c.V` with it.

A selection sort, because it needs no permutation vector and no temporary column: the rank is
read as a *leading run* of `c.S` and [`ldiv!`](@ref) indexes `1:rank` as a leading block, so
the order is load-bearing rather than cosmetic, and it has to be established without allocating.
"""
function _sort_singular_values!(c::SVDCache)
    n = length(c.S)
    for j in 1:(n - 1)
        k = j
        for i in (j + 1):n
            c.S[i] > c.S[k] && (k = i)
        end
        k == j && continue
        c.S[j], c.S[k] = c.S[k], c.S[j]
        @inbounds for i in 1:n
            c.A[i, j], c.A[i, k] = c.A[i, k], c.A[i, j]
            c.V[i, j], c.V[i, k] = c.V[i, k], c.V[i, j]
        end
    end
    c
end

"""
    _decompose!(method::SVDSolver, c::SVDCache)

Run the one-sided Jacobi iteration on `c.A` to convergence, then read off the singular values,
order them, determine the numerical rank and normalize `c.A` into `U`.

# The columns beyond the rank

They are set to zero rather than completed into an orthonormal basis. ``A = U\\Sigma V^*``
stays exact — a zero column paired against a zero singular value contributes nothing — and
[`ldiv!`](@ref) never reads them, because the truncation zeroes those coordinates before the
second product. What is given up is ``U^* U = I``, which no caller can observe: `U` is not
exposed by any accessor. Completing the basis would be work in service of an invariant nothing
checks. LAPACK's `gesdd` does return an orthogonal `U`, so the difference is worth knowing
about when comparing the two caches directly.
"""
function _decompose!(method::SVDSolver, c::SVDCache{T, RT}) where {T, RT}
    n = size(c.A, 1)
    scale = c.scale = _prescale!(c.A)

    fill!(c.V, zero(T))
    @inbounds for i in 1:n
        c.V[i, i] = one(T)
    end

    # Every rotation is rounded back into `T`, so the sweeps bottom out at a measured
    # `0.07 · √n · eps(T)` — the same multiple for `Float16`, `Float32` and `Float64`, and
    # reached in six to ten sweeps. This threshold sits an order above that floor: asking for
    # the floor itself would risk a matrix that approaches it without ever going under.
    tol = sqrt(RT(n)) * eps(RT)
    sweeps = 0
    while _jacobi_sweep!(c.A, c.V, tol) > 0
        sweeps += 1
        sweeps < JACOBI_MAX_SWEEPS || error(
            "the one-sided Jacobi iteration did not orthogonalize the columns of a " *
            "$(n) × $(n) $(T) matrix within $(JACOBI_MAX_SWEEPS) sweeps; the singular " *
            "values it would report are not converged, and a rank read off them would not " *
            "mean what it says")
    end

    for j in 1:n
        c.S[j] = _acc_norm(view(c.A, :, j))
    end
    _sort_singular_values!(c)
    c.rank = _rank_from_decreasing(c.S, rank_tolerance(method, T))

    # `A` becomes `U`. Beyond the rank the columns are zeroed rather than normalized — see
    # above for why that is the right answer and not a shortcut — which also keeps a division
    # by a singular value that is rounding noise from leaking an `Inf` into the solve.
    for j in 1:n
        if j ≤ c.rank
            σ = c.S[j]
            @inbounds for i in 1:n
                c.A[i, j] /= σ
            end
        else
            @inbounds for i in 1:n
                c.A[i, j] = zero(T)
            end
        end
    end

    # `_prescale!` scaled the matrix, and a scaled matrix has scaled singular values. `U` and
    # `V` are unaffected by a positive scalar.
    c.S ./= scale
    c
end

function singular_values(lsolver::LinearSolver{T, LSM}) where {T, LSM <: SVDSolver}
    checkfactorized(lsolver)
    cache(lsolver).S
end

function LinearAlgebra.ldiv!(x::AbstractVector{T}, lsolver::LinearSolver{T, LSM},
        b::AbstractVector{T}) where {T, LSM <: SVDSolver}
    c = cache(lsolver)
    checkfactorized(lsolver)
    @assert axes(x, 1) == axes(b, 1) == axes(c.A, 1)
    Base.require_one_based_indexing(x, b)

    n = size(c.A, 2)
    r = c.rank

    # `b` is consumed into `y` before anything is written to `x`, so `x === b` is fine, and
    # `A` holds `U` once `factorize!` has run
    mul!(c.y, adjoint(c.A), b)

    # the truncated pseudo-inverse. Zeroing rather than dividing beyond the rank is the whole
    # of the minimum-norm property: those are the coordinates along the null space, and
    # dividing by a singular value that is rounding noise is what produced the enormous steps
    # this method exists to avoid.
    #
    # `c.S` holds the singular values of the caller's matrix, so `c.S * c.scale` recovers those
    # of the scaled one — exactly, `c.scale` being a power of two. Dividing by those instead
    # keeps the quotient below `‖b‖ √n / rank_tolerance`, where dividing by `c.S` directly
    # overflows for a matrix small enough that `c.scale` is large: in `Float16` the quotient
    # can exceed `floatmax` while the solution itself is representable.
    for i in 1:n
        c.y[i] = i ≤ r ? c.y[i] / (c.S[i] * c.scale) : zero(T)
    end

    # and `x` carries the factor the division no longer does, as it does for `PivotedQR`: the
    # `x` that solves `sAx = b` is `1/s` times the one that solves `Ax = b`.
    mul!(x, c.V, c.y)
    isone(c.scale) || (x .*= c.scale)
    x
end
