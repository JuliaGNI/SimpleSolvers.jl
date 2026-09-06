"""
    struct SVDSolver{RT} <: RankRevealingMethod{RT}

A singular value decomposition, meant to solve a [`LinearProblem`](@ref) whose matrix is rank
deficient — and to say how deficient it is.

Like [`PivotedQR`](@ref) it returns the **minimum-norm solution** rather than raising a
`SingularException`; unlike it, the decomposition it computes is the one the question is
usually asked in. [`singular_values`](@ref) hands back the spectrum the rank was read off, so
"is this matrix rank deficient, or merely ill-conditioned?" is answered by the same call that
solved the system — a gap of several orders in that spectrum is the first, a smooth decay the
second, and no rank tolerance can tell them apart on its own.

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

`gesdd` — the divide-and-conquer driver — factorizes ``A = U \\Sigma V^*``. The numerical rank
is the number of leading singular values above the tolerance, and [`ldiv!`](@ref) applies the
truncated pseudo-inverse ``V \\Sigma_r^{-1} U^*``, which is the minimum-norm least-squares
solution by construction.

# Cost

The decomposition is the expensive one — 2–4× [`PivotedQR`](@ref)'s, and it allocates heavily,
because LAPACK's Julia wrapper returns freshly allocated factors and there is no
preallocated-output `gesdd` to point a cache at. The *solve* is the cheap one: two `gemv`s
against a factorization already in the cache, allocating nothing, where
[`PivotedQR`](@ref)'s applies two blocked orthogonal factors to a single column and allocates
its own LAPACK workspace each time.

So the sum favours [`PivotedQR`](@ref) at one solve per factorization — a Newton step — and
this method when a factorization is reused across several right-hand sides, when per-solve
allocation is what matters, or when the spectrum is part of what is wanted. The measured table
is in [`PivotedQR`](@ref)'s docstring.

Neither is the default for anything — see [`default_linear_solver_method`](@ref).

Restricted to the element types LAPACK provides (`Float32`, `Float64`, `ComplexF32` and
`ComplexF64`), and to square matrices, exactly as [`PivotedQR`](@ref) is.

# Example

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: singular_values; using LinearAlgebra: rank, ldiv!)
julia> A = [1.0 2.0; 2.0 4.0];           # rank 1

julia> ls = LinearSolver(SVDSolver(), A);

julia> factorize!(ls, A);

julia> rank(ls)
1

julia> round.(singular_values(ls); digits = 10)
2-element Vector{Float64}:
 5.0
 0.0

julia> round.(ldiv!(zeros(2), ls, [1.0, 2.0]); digits = 10)
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
- `A`: the working copy of the matrix, which `gesdd` destroys,
- `U`, `S`, `Vt`: the factors ``A = U \\Sigma V^*``, with `S` real even for a complex matrix,
- `y`: an `n`-vector holding ``U^* b`` while it is scaled, so that [`ldiv!`](@ref) allocates
  nothing and tolerates `x === b`,
- `rank`: the numerical rank [`factorize!`](@ref) found,
- `factorized`: whether [`factorize!`](@ref) has run at all, which `rank == 0` cannot express.

The three factors are *reassigned* by each [`factorize!`](@ref) rather than written into:
LAPACK's Julia wrapper allocates and returns them, and there is no variant that fills arrays
the cache already owns. They are allocated once at construction all the same, so that the
cache is a valid object — with a rank of zero and `factorized` false — before anything is
factorized.
"""
mutable struct SVDCache{T, RT <: Real, AT <: AbstractMatrix{T}} <: LinearSolverCache{T}
    A::AT
    U::Matrix{T}
    S::Vector{RT}
    Vt::Matrix{T}
    y::Vector{T}
    rank::Int
    factorized::Bool
end

function LinearSolverCache(method::SVDSolver, A::AbstractMatrix{T}) where {T}
    _blas_eltype_check(method, T)
    n = checksquare(A)
    Ā = Matrix{T}(A)
    SVDCache{T, real(T), typeof(Ā)}(Ā, zeros(T, n, n), zeros(real(T), n), zeros(T, n, n),
        zeros(T, n), 0, false)
end

function _decompose!(method::SVDSolver, c::SVDCache{T}) where {T}
    # `'A'` and not `'S'`: the two agree for a square matrix, and `'A'` is what makes `U` and
    # `Vt` the square factors the docstring names.
    c.U, c.S, c.Vt = LinearAlgebra.LAPACK.gesdd!('A', c.A)
    c.rank = _rank_from_decreasing(c.S, rank_tolerance(method, T))
    c
end

"""
    singular_values(lsolver::LinearSolver{T,<:SVDSolver})

The singular values of the matrix that was factorized, in non-increasing order.

The array is the cache's own, so it is overwritten by the next [`factorize!`](@ref) — copy it
if it has to outlive one. There is no counterpart for [`PivotedQR`](@ref): the moduli of its
`R` diagonal bound the singular values but are not equal to them, and reporting them under
this name would invite reading a spectrum off numbers that are not one.
"""
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

    # `b` is consumed into `y` before anything is written to `x`, so `x === b` is fine
    mul!(c.y, adjoint(c.U), b)

    # the truncated pseudo-inverse. Zeroing rather than dividing beyond the rank is the whole
    # of the minimum-norm property: those are the coordinates along the null space, and
    # dividing by a singular value that is rounding noise is what produced the enormous steps
    # this method exists to avoid.
    for i in 1:n
        c.y[i] = i ≤ r ? c.y[i] / c.S[i] : zero(T)
    end

    mul!(x, adjoint(c.Vt), c.y)
    x
end
