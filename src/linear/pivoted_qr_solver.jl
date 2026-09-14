"""
    struct PivotedQR{RT} <: RankRevealingMethod{RT}

A complete orthogonal factorization written in plain Julia, meant to solve a
[`LinearProblem`](@ref) whose matrix is rank deficient.

Where every [`PivotedLUMethod`](@ref) answers a singular matrix with a `SingularException`,
this one answers it with the **minimum-norm solution**: of all the `x` that minimize
``\\lVert Ax - b \\rVert``, the one with the smallest ``\\lVert x \\rVert``. On a *consistent*
system — one whose right-hand side lies in the range of `A`, as a Newton residual at a
critical point does — that is an exact solution of ``Ax = b`` with no component along the null
space at all.

[`LapackPivotedQR`](@ref) computes the same thing through `geqp3` and `tzrzf` and is the
faster of the two on the four element types LAPACK has. This one is the counterpart that
takes **any floating-point element type**, `Float16` and `BigFloat` included — the same
relationship [`LU`](@ref) has to [`LapackLU`](@ref).

# Constructor

```jldoctest; setup = :(using SimpleSolvers)
PivotedQR()

# output

PivotedQR{Missing}(missing)
```

`rtol` is the relative threshold that separates a direction the matrix has from one it does
not; `missing` resolves it from the element type. See [`rank_tolerance`](@ref), which is where
the default and its justification live.

```jldoctest; setup = :(using SimpleSolvers)
PivotedQR(; rtol = 1e-10)

# output

PivotedQR{Float64}(1.0e-10)
```

# What it computes

A Businger–Golub column-pivoted `QR`, ``AP = QR``, with the columns ordered so that
``\\lvert R_{ii} \\rvert`` is non-increasing; the numerical rank `r` is the length of the
leading run above the tolerance. When `r < n` the leading `r` rows are then reduced to
``\\begin{pmatrix} L & 0 \\end{pmatrix} Z`` with `L` lower triangular and `Z` orthogonal,
giving a *complete orthogonal* factorization — see [`_decompose!`](@ref) for how that second
step is obtained from a second `QR` rather than from a factorization of its own.

!!! warning "A column-pivoted QR on its own is not enough"
    Stopping after the pivoted `QR` and solving with the leading `r` columns gives a *basic*
    solution — zeros in the columns pivoting dropped — not the minimum-norm one, and on a
    matrix with a wide null space the two differ by a great deal. The `Z` factor is exactly
    what removes the null-space component, so it is not an optimization that can be skipped.

# Cost

The pivot search recomputes the trailing column norms exactly at every step, which is
``O(n^2)`` work per step rather than the ``O(n)`` of LAPACK's downdated running norms. That
roughly doubles the flop count, and it buys something: on a Kahan matrix — the classical case
constructed to defeat column pivoting — the exact norms keep the pivot order that exposes the
small direction where the downdated ones drift past it, so this method reports the rank the
spectrum has and [`LapackPivotedQR`](@ref) reports one too many. The table is in
[`rank`](@ref).

**Both [`factorize!`](@ref) and [`ldiv!`](@ref) are allocation-free**, which neither
[`LapackPivotedQR`](@ref) nor [`LapackSVDSolver`](@ref) is: LAPACK's Julia wrappers allocate
their own factors and workspace, and `ormrz` alone asks for 98496 bytes on every solve. That
holds for any `isbitstype` element type; `BigFloat` allocates per arithmetic operation and
nothing here can change that.

No rank-revealing method is the default for anything — see
[`default_linear_solver_method`](@ref). A singular matrix is a bug in most callers, and a
method that quietly returns a minimum-norm step everywhere would hide it.

Square matrices only: the rectangular least-squares problem this decomposition also solves is
not exposed here, because the rest of the [`LinearSolver`](@ref) interface — the `solve!`
forms, `alloc_rhs`, the vector constructor — is built around one dimension.

# Example

```jldoctest; setup = :(using SimpleSolvers)
julia> A = Float16[1 2; 2 4];            # rank 1

julia> b = Float16[1, 2];                # consistent: b = A * [0.2, 0.4]

julia> x = solve(PivotedQR(), A, b);

julia> round.(Float64.(x); digits = 2)   # the minimum-norm solution, not a basic one
2-element Vector{Float64}:
 0.2
 0.4
```
"""
struct PivotedQR{RT} <: RankRevealingMethod{RT}
    rtol::RT

    PivotedQR(; rtol = missing) = new{typeof(rtol)}(rtol)
end

"""
    PivotedQRCache <: LinearSolverCache

The cache of a [`PivotedQR`](@ref).

# Keys
- `A`: the working copy of the matrix, overwritten with `R` in the upper triangle and the
  reflector tails of `Q` below it,
- `Z`: the second factorization, of the adjoint of `A`'s leading `rank` rows; untouched while
  the matrix has full column rank,
- `tau`, `ztau`: the Householder scalars of the two factorizations,
- `jpvt`: the column permutation, so that `A[:, jpvt] = Q * R`,
- `y`: an `n`-vector holding the right-hand side while it is transformed, so that
  [`ldiv!`](@ref) allocates nothing and tolerates `x === b`,
- `scale`: the power of two [`_prescale!`](@ref) multiplied the matrix by,
- `rank`: the numerical rank [`factorize!`](@ref) found,
- `factorized`: whether [`factorize!`](@ref) has run at all, which `rank == 0` cannot express.

Every buffer is allocated once, at construction, and written into thereafter — which is what
makes both [`factorize!`](@ref) and [`ldiv!`](@ref) allocation-free. `Z` costs a second `n × n`
array and is what pays for the second factorization being an ordinary `QR`.

`jpvt` is a `Vector{Int}` and not the `Vector{LinearAlgebra.BlasInt}` of
[`LapackPivotedQRCache`](@ref): nothing here is handed to LAPACK, so there is no 32-bit-integer
BLAS to match.
"""
mutable struct PivotedQRCache{T, RT <: Real, AT <: AbstractMatrix{T}} <:
               LinearSolverCache{T}
    A::AT
    Z::Matrix{T}
    tau::Vector{T}
    ztau::Vector{T}
    jpvt::Vector{Int}
    y::Vector{T}
    scale::RT
    rank::Int
    factorized::Bool
end

function LinearSolverCache(method::PivotedQR, A::AbstractMatrix{T}) where {T}
    _float_eltype_check(method, T)
    n = checksquare(A)
    Ā = Matrix{T}(A)
    PivotedQRCache{T, real(T), typeof(Ā)}(Ā, zeros(T, n, n), zeros(T, n), zeros(T, n),
        zeros(Int, n), zeros(T, n), one(real(T)), 0, false)
end

"""
    _reflector!(x) -> τ

Overwrite `x` with the Householder reflector that maps it onto its first coordinate axis, and
return the scalar `τ`.

The convention is LAPACK's `larfg`: the reflector is ``H = I - \\tau v v^*`` with
``v_1 = 1`` *implicit* — on return `x[1]` holds the resulting diagonal entry ``-\\nu`` rather
than that one, and `x[2:end]` holds the tail of `v`. The sign ``\\nu = \\pm\\lVert x\\rVert``
follows that of `x[1]`, which is what keeps ``x_1 + \\nu`` from cancelling and bounds the tail
by one in modulus. A zero `x` returns `τ = 0`, i.e. `H = I`.

Written here rather than taken from `LinearAlgebra.reflector!`, which is internal and
undocumented and whose body differs between the Julia versions this package supports.
"""
function _reflector!(x::AbstractVector{T}) where {T}
    ξ = @inbounds x[1]
    ν = convert(real(T), copysign(_acc_norm(x), real(ξ)))
    iszero(ν) && return zero(T)
    ξ += ν
    @inbounds x[1] = -ν
    @inbounds for i in 2:length(x)
        x[i] /= ξ
    end
    ξ / ν
end

"""
    _reflect!(v, c, y)
    _reflect!(v, c, A)

Apply ``I - c\\,v v^*`` in place to a vector, or to every column of a matrix.

`v` is a reflector as [`_reflector!`](@ref) left it, so **`v[1]` is never read** — it is taken
to be one. `y`, or each column of `A`, has the same length as `v`; the loops are `@inbounds`
and do not check it. Which of `H` and ``H^*`` this is follows from `c` rather than from a flag: pass
`conj(τ)` for ``H^*``, which is what a factorization applies, and `τ` for `H`, which is what
reconstructing the orthogonal factor applies.
"""
function _reflect!(v::AbstractVector, c, y::AbstractVector)
    iszero(c) && return y
    n = length(v)
    s = @inbounds y[1]
    @inbounds for i in 2:n
        s += conj(v[i]) * y[i]
    end
    s *= c
    @inbounds y[1] -= s
    @inbounds for i in 2:n
        y[i] -= s * v[i]
    end
    y
end

function _reflect!(v::AbstractVector, c, A::AbstractMatrix)
    for j in axes(A, 2)
        _reflect!(v, c, view(A, :, j))
    end
    A
end

"""
    _maxnorm_column(A, k) -> Int

The index of the column of `A` whose rows `k:end` have the largest norm, searching from `k`.

The Businger–Golub pivot choice, computed from exact norms at every step. `LinearAlgebra`'s own
generic pivoted `QR` does the same; LAPACK instead carries running norms and downdates them,
which is asymptotically cheaper and needs a cancellation safeguard to stay correct. See
[`PivotedQR`](@ref) for why the exact form is the one here.
"""
function _maxnorm_column(A::AbstractMatrix, k::Integer)
    m = size(A, 1)
    p = k
    best = zero(real(_accumulator(eltype(A))))
    for j in k:size(A, 2)
        nrm = _acc_norm(view(A, k:m, j))
        if nrm > best
            p, best = j, nrm
        end
    end
    p
end

"""
    _swap_columns!(A, jpvt, j, k)

Exchange columns `j` and `k` of `A` and the corresponding entries of `jpvt`. A no-op when the
two indices coincide, so the caller need not test for it.
"""
function _swap_columns!(A::AbstractMatrix, jpvt::AbstractVector{Int}, j::Integer, k::Integer)
    j == k && return A
    jpvt[j], jpvt[k] = jpvt[k], jpvt[j]
    @inbounds for i in axes(A, 1)
        A[i, j], A[i, k] = A[i, k], A[i, j]
    end
    A
end

"""
    _qr!(A, tau)

Householder `QR` of `A` in place, without pivoting, for a matrix at least as tall as it is
wide. `R` ends in the upper triangle, the reflector tails below it and their scalars in `tau`.
"""
function _qr!(A::AbstractMatrix, tau::AbstractVector)
    m, n = size(A)
    for j in 1:n
        v = view(A, j:m, j)
        tau[j] = _reflector!(v)
        _reflect!(v, conj(tau[j]), view(A, j:m, (j + 1):n))
    end
    A
end

"""
    _pivoted_qr!(A, tau, jpvt)

Businger–Golub column-pivoted Householder `QR` of the square matrix `A` in place, so that
`A[:, jpvt] = Q * R` and ``\\lvert R_{ii}\\rvert`` is non-increasing.
"""
function _pivoted_qr!(A::AbstractMatrix, tau::AbstractVector, jpvt::AbstractVector{Int})
    n = size(A, 1)
    jpvt .= 1:n
    for k in 1:n
        _swap_columns!(A, jpvt, k, _maxnorm_column(A, k))
        v = view(A, k:n, k)
        tau[k] = _reflector!(v)
        _reflect!(v, conj(tau[k]), view(A, k:n, (k + 1):n))
    end
    A
end

"""
    _solve_upper!(y, R, r)

Overwrite `y[1:r]` with the solution of ``R_{1:r,1:r}\\,w = y_{1:r}`` by back substitution.
"""
function _solve_upper!(y::AbstractVector, R::AbstractMatrix, r::Integer)
    @inbounds for i in r:-1:1
        s = y[i]
        for j in (i + 1):r
            s -= R[i, j] * y[j]
        end
        y[i] = s / R[i, i]
    end
    y
end

"""
    _solve_lower_adjoint!(y, Z, r)

Overwrite `y[1:r]` with the solution of ``L w = y_{1:r}``, where ``L = R^*`` is the lower
triangle implied by the upper triangle `R` sitting in `Z[1:r, 1:r]`. Forward substitution, with
the conjugation applied entry by entry so that `L` never has to be formed.
"""
function _solve_lower_adjoint!(y::AbstractVector, Z::AbstractMatrix, r::Integer)
    @inbounds for i in 1:r
        s = y[i]
        for j in 1:(i - 1)
            s -= conj(Z[j, i]) * y[j]
        end
        y[i] = s / conj(Z[i, i])
    end
    y
end

"""
    _decompose!(method::PivotedQR, c::PivotedQRCache)

Compute the complete orthogonal factorization of `c.A` in place and record the numerical rank.

# The second factorization

The pivoted `QR` leaves ``AP = Q\\begin{pmatrix} B \\\\ 0\\end{pmatrix}`` with `B` the upper
trapezoidal leading `r × n` block. Writing its adjoint as an ordinary `QR`,
``B^* = Q_Z R_Z``, gives ``B = R_Z^* Q_Z^*``, so with ``Z := Q_Z^*`` the block reads
``\\begin{pmatrix} L & 0\\end{pmatrix} Z`` with ``L = R_Z^*`` lower triangular — a complete
orthogonal factorization, `L` lower where LAPACK's `tzrzf` produces an upper `T`.

Obtaining it this way means there is no second kind of reflector to generate or apply: `Z` is
an ordinary Householder factor, produced by [`_qr!`](@ref) and consumed by the same
[`_reflect!`](@ref) the first factorization uses. The `RZ` reflectors LAPACK stores instead
have the shape ``[1, 0, \\ldots, 0, z]`` and need a generate and an apply of their own.
"""
function _decompose!(method::PivotedQR, c::PivotedQRCache{T}) where {T}
    n = size(c.A, 1)
    c.scale = _prescale!(c.A)
    _pivoted_qr!(c.A, c.tau, c.jpvt)

    # column pivoting orders the columns so that |R_ii| is non-increasing, so the rank is a
    # leading run — see `_rank_from_decreasing`, which takes the diagonal as a generator rather
    # than asking for a `diag` vector.
    c.rank = _rank_from_decreasing(
        (abs(c.A[i, i]) for i in 1:n), rank_tolerance(method, T))

    # There is nothing to annihilate at full column rank, and nothing to annihilate in a matrix
    # that is numerically zero either.
    r = c.rank
    if 0 < r < n
        # `A`'s leading `r` rows carry `R`'s trapezoid on and above the diagonal and `Q`'s
        # reflector tails below it, so the adjoint is taken triangle by triangle rather than
        # row by row.
        @inbounds for i in 1:r, j in 1:n

            c.Z[j, i] = j < i ? zero(T) : conj(c.A[i, j])
        end
        _qr!(view(c.Z, :, 1:r), c.ztau)
    end
    c
end

function LinearAlgebra.ldiv!(x::AbstractVector{T}, lsolver::LinearSolver{T, LSM},
        b::AbstractVector{T}) where {T, LSM <: PivotedQR}
    c = cache(lsolver)
    checkfactorized(lsolver)
    @assert axes(x, 1) == axes(b, 1) == axes(c.A, 1)
    Base.require_one_based_indexing(x, b)

    n = size(c.A, 1)
    r = c.rank

    # a numerically zero matrix has only the zero solution to minimize over
    if r == 0
        fill!(x, zero(T))
        return x
    end

    # the transformations are in place and `x === b` is allowed, so the right-hand side moves
    # into the cache's working vector first and `x` is written only at the very end
    copyto!(c.y, b)

    for k in 1:n
        _reflect!(view(c.A, k:n, k), conj(c.tau[k]), view(c.y, k:n))
    end

    if r == n
        _solve_upper!(c.y, c.A, n)
    else
        _solve_lower_adjoint!(c.y, c.Z, r)

        # Zeroing the trailing entries is what selects the minimum-norm solution out of the
        # affine set of solutions: they are the coordinates along the null space, and `Z` below
        # carries them back as a component of `x` if they are left in.
        for i in (r + 1):n
            c.y[i] = zero(T)
        end

        # y ← Zᴴ y, and Zᴴ is Q_Z itself — so the reflectors go in reverse order and their
        # scalars are applied unconjugated, the mirror of the loop above.
        for k in r:-1:1
            _reflect!(view(c.Z, k:n, k), c.ztau[k], view(c.y, k:n))
        end
    end

    # `_prescale!` multiplied the matrix by `scale`, so what `c.y` holds solves
    # `scale * A * x = b` and is `1/scale` times the solution of `A * x = b`.
    for i in 1:n
        x[c.jpvt[i]] = c.scale * c.y[i]
    end
    x
end
