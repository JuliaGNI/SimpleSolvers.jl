"""
    struct PivotedQR{RT} <: RankRevealingMethod{RT}

A complete orthogonal factorization, meant to solve a [`LinearProblem`](@ref) whose matrix is
rank deficient.

Where every [`PivotedLUMethod`](@ref) answers a singular matrix with a `SingularException`,
this one answers it with the **minimum-norm solution**: of all the `x` that minimize
``\\lVert Ax - b \\rVert``, the one with the smallest ``\\lVert x \\rVert``. On a *consistent*
system — one whose right-hand side lies in the range of `A`, as a Newton residual at a
critical point does — that is an exact solution of ``Ax = b`` with no component along the null
space at all.

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

`geqp3` factorizes ``AP = QR`` with the columns ordered so that ``\\lvert R_{ii} \\rvert`` is
non-increasing; the numerical rank `r` is the length of the leading run above the tolerance.
When `r < n` the trailing columns of the leading `r` rows are then annihilated by `tzrzf`,
giving ``AP = Q \\begin{pmatrix} T & 0 \\\\ 0 & 0\\end{pmatrix} Z`` with `T` upper triangular
and `Q`, `Z` orthogonal — a *complete orthogonal* factorization. This is the decomposition
LAPACK's `gelsy` computes, split into a [`factorize!`](@ref) and an [`ldiv!`](@ref) so that
one factorization can serve several right-hand sides.

!!! warning "A column-pivoted QR on its own is not enough"
    Stopping after `geqp3` and solving with the leading `r` columns gives a *basic* solution —
    zeros in the columns pivoting dropped — not the minimum-norm one, and on a matrix with a
    wide null space the two differ by a great deal. The `Z` factor is exactly what removes the
    null-space component, so it is not an optimization that can be skipped.

# When to use which

[`SVDSolver`](@ref) computes the same minimum-norm solution and additionally reports the
singular values. This method is the cheaper of the two *per factorization*, and the more
expensive per solve. Measured on an Apple M4 Max against OpenBLAS, on a matrix of rank
``\\lfloor n/2 \\rfloor``, times in microseconds and allocation in bytes for a
[`LinearSolver`](@ref) that has already been built — `scripts/benchmark_rank_revealing.jl`:

| n | | `factorize!` | | `ldiv!` | |
|---:|:---|---:|---:|---:|---:|
| | | µs | B | µs | B |
| 13 | [`LapackLU`](@ref)\\* | 0.88 | 0 | 0.12 | 0 |
| | `PivotedQR` | **1.83** | 6000 | 0.71 | 98496 |
| | [`SVDSolver`](@ref) | 8.12 | 11392 | **0.08** | **0** |
| 40 | [`LapackLU`](@ref)\\* | 5.46 | 0 | 0.58 | 0 |
| | `PivotedQR` | **19.6** | 17824 | 8.83 | 98496 |
| | [`SVDSolver`](@ref) | 72.5 | 81072 | **0.38** | **0** |
| 128 | [`LapackLU`](@ref)\\* | 64.5 | 0 | 3.12 | 0 |
| | `PivotedQR` | **352** | 66304 | 81.5 | 98496 |
| | [`SVDSolver`](@ref) | 1100 | 681344 | **4.04** | **0** |
| 384 | [`LapackLU`](@ref)\\* | 601 | 0 | 25.5 | 0 |
| | `PivotedQR` | **7201** | 165696 | 475 | 98496 |
| | [`SVDSolver`](@ref) | 16934 | 5959008 | **35.0** | **0** |

\\* [`LapackLU`](@ref) is timed on a *full-rank* matrix of the same size, since it throws a
`SingularException` on the deficient one. Its row is a scale, not a comparison: a
rank-revealing decomposition costs several times an LU, and that is the price of the answer.

Two things to read off, both of which cut against the obvious expectation.

**Neither method's `factorize!` is allocation-free**, unlike every [`PivotedLUMethod`](@ref)
here. LAPACK's Julia wrappers allocate and return their factors and their workspace, and there
is no preallocated-output variant of `geqp3`, `tzrzf` or `gesdd` to point a cache at.

**This method's `ldiv!` is the slower and the allocating one**, by an order of magnitude at
`n = 384`. `ormqr` and `ormrz` apply a blocked orthogonal factor to a single right-hand-side
column, which is the case blocking is worst at, and each queries and allocates its own
workspace — 98496 bytes, sized by the BLAS blocking factor rather than by `n`, which is why
that number does not move with the matrix. [`SVDSolver`](@ref)'s solve is two `gemv`s and
allocates nothing.

So the choice follows the ratio of solves to factorizations. **One solve per factorization —
a Newton step — is this method**, which wins on the sum by 3.2× at `n = 13` and 2.2–2.7× over
the rest of the range above. Many solves
against one factorization, or a hot loop where per-solve allocation is what matters, is
[`SVDSolver`](@ref), as is any question about the spectrum itself.

Neither is the default for anything — see [`default_linear_solver_method`](@ref). A singular
matrix is a bug in most callers, and a method that quietly returns a minimum-norm step
everywhere would hide it.

Restricted to the element types LAPACK provides (`Float32`, `Float64`, `ComplexF32` and
`ComplexF64`); anything else raises an `ArgumentError` naming the type. Square matrices only:
the rectangular least-squares problem this decomposition also solves is not exposed here,
because the rest of the [`LinearSolver`](@ref) interface — the `solve!` forms, `alloc_rhs`,
the vector constructor — is built around one dimension.

# Example

```jldoctest; setup = :(using SimpleSolvers)
julia> A = [1.0 2.0; 2.0 4.0];           # rank 1

julia> b = [1.0, 2.0];                   # consistent: b = A * [0.2, 0.4]

julia> x = solve(PivotedQR(), A, b);

julia> round.(x; digits = 10)            # the minimum-norm solution, not a basic one
2-element Vector{Float64}:
 0.2
 0.4

julia> A * x ≈ b
true
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
- `A`: the working copy of the matrix, overwritten with the `QR` factors and then with the
  `RZ` factors of its leading `rank` rows,
- `jpvt`: the column permutation `geqp3` chose,
- `tau`: the reflectors of `Q`,
- `tzt`: the reflectors of `Z`, empty while the matrix has full column rank,
- `C`: an `n × 1` working right-hand side — LAPACK's `ormqr`/`ormrz` transform a *matrix*,
- `rank`: the numerical rank [`factorize!`](@ref) found,
- `factorized`: whether [`factorize!`](@ref) has run at all, which `rank == 0` cannot express.

`tzt` is a field rather than a local because `ormrz` needs it again at solve time, and it is
reassigned rather than written into: `tzrzf!` allocates its own reflector vector. Neither
[`factorize!`](@ref) nor [`ldiv!`](@ref) is allocation-free here the way a
[`PivotedLUMethod`](@ref)'s is — see the table in [`PivotedQR`](@ref) for what that costs and
where it comes from. `C` and the cached arrays keep the allocation to LAPACK's own workspace
rather than adding the package's on top.

The pivot vector is typed `Vector{LinearAlgebra.BlasInt}` for the same reason
[`PivotedLUCache`](@ref)'s is: that is what LAPACK fills, and it is not `Vector{Int}` under a
32-bit-integer BLAS.
"""
mutable struct PivotedQRCache{T, AT <: AbstractMatrix{T}} <: LinearSolverCache{T}
    A::AT
    jpvt::Vector{LinearAlgebra.BlasInt}
    tau::Vector{T}
    tzt::Vector{T}
    C::Matrix{T}
    rank::Int
    factorized::Bool
end

function LinearSolverCache(method::PivotedQR, A::AbstractMatrix{T}) where {T}
    _blas_eltype_check(method, T)
    n = checksquare(A)
    Ā = Matrix{T}(A)
    PivotedQRCache{T, typeof(Ā)}(Ā, zeros(LinearAlgebra.BlasInt, n), Vector{T}(undef, n),
        T[], zeros(T, n, 1), 0, false)
end

function _decompose!(method::PivotedQR, c::PivotedQRCache{T}) where {T}
    n = size(c.A, 2)
    fill!(c.jpvt, 0)                     # 0 = every column is free to be pivoted
    LinearAlgebra.LAPACK.geqp3!(c.A, c.jpvt, c.tau)

    # `geqp3` orders the columns so that |R_ii| is non-increasing, so the rank is a leading
    # run — see `_rank_from_decreasing`, which takes the diagonal as a generator rather than
    # asking for a `diag` vector.
    c.rank = _rank_from_decreasing(
        (abs(c.A[i, i]) for i in 1:n), rank_tolerance(method, T))

    # The complete orthogonal step: annihilate the trailing columns of the leading `rank`
    # rows, so that the triangular solve below is followed by an orthogonal map back into the
    # full column space. There is nothing to annihilate at full column rank — and `tzt` is
    # emptied rather than left alone, so that a stale set of reflectors from a previous
    # factorization cannot be reached.
    if 0 < c.rank < n
        _, c.tzt = LinearAlgebra.LAPACK.tzrzf!(view(c.A, 1:(c.rank), :))
    else
        c.tzt = T[]
    end
    c
end

function LinearAlgebra.ldiv!(x::AbstractVector{T}, lsolver::LinearSolver{T, LSM},
        b::AbstractVector{T}) where {T, LSM <: PivotedQR}
    c = cache(lsolver)
    checkfactorized(lsolver)
    @assert axes(x, 1) == axes(b, 1) == axes(c.A, 1)
    Base.require_one_based_indexing(x, b)

    n = size(c.A, 2)
    r = c.rank

    # a numerically zero matrix has only the zero solution to minimize over
    if r == 0
        fill!(x, zero(T))
        return x
    end

    # the transformations are in place and `x === b` is allowed, so the right-hand side moves
    # into the cache's working column first and `x` is written only at the very end
    copyto!(c.C, b)

    LinearAlgebra.LAPACK.ormqr!('L', _adjoint_char(T), c.A, c.tau, c.C)
    LinearAlgebra.LAPACK.trtrs!('U', 'N', 'N', view(c.A, 1:r, 1:r), view(c.C, 1:r, :))

    # Zeroing the trailing entries is what selects the minimum-norm solution out of the affine
    # set of solutions: they are the coordinates along the null space, and `Z` below carries
    # them back as a component of `x` if they are left in.
    for i in (r + 1):n
        c.C[i, 1] = zero(T)
    end
    r < n &&
        LinearAlgebra.LAPACK.ormrz!('L', _adjoint_char(T), view(c.A, 1:r, :), c.tzt, c.C)

    for i in 1:n
        x[c.jpvt[i]] = c.C[i, 1]
    end
    x
end
