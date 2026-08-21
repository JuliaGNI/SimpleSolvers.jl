"""
    struct LapackLU <: DirectMethod

A LAPACK-backed LU solver, meant to solve a [`LinearProblem`](@ref).

Where [`LU`](@ref) is a self-contained implementation that works for any number type and for
static matrices, this method delegates the factorization to `LinearAlgebra.lu!` and therefore
to LAPACK. It is restricted to the element types LAPACK handles — `Float32`, `Float64`,
`ComplexF32` and `ComplexF64` — and throws an `ArgumentError` naming the type when handed
anything else. Any `AbstractMatrix` storage is accepted, but the cache always holds a plain
`Matrix`, because that is what LAPACK can be pointed at.

Routines that use it are the same as for [`LU`](@ref): [`factorize!`](@ref), [`ldiv!`](@ref)
and [`solve!`](@ref).

# Constructor

```jldoctest; setup = :(using SimpleSolvers)
LapackLU()

# output

LapackLU()
```

# When to use which

[`LU`](@ref) is the better choice for small systems, where its static-matrix cache avoids
allocation altogether, and it is the only choice for element types LAPACK does not know
about. `LapackLU` is the better choice once the systems are large enough for the
``\\mathcal{O}(n^3)`` factorization to dominate, because a blocked LAPACK kernel is a great
deal faster than a scalar loop there.

The crossover is problem-dependent, but the effect is large: measured from
[PoissonBrackets.jl](https://github.com/JuliaGNI/PoissonBrackets.jl), where a Newton step
factorizes a dense ``384 \\times 384`` Jacobian, [`LU`](@ref) accounted for 74 % of the cost
of one implicit time step — about 17 ms against 0.6 ms for the same factorization through
LAPACK.

Allocation is *not* part of the trade-off. Like [`LU`](@ref), and unlike a bare
`LinearAlgebra.lu!`, both [`factorize!`](@ref) and [`ldiv!`](@ref) are allocation-free after
the [`LinearSolver`](@ref) is built: the working matrix and the pivot vector are allocated
once and reused. See [`LapackLUSolverCache`](@ref).

The one exception is [`factorize!`](@ref) on a Julia too old for
`LAPACK.getrf!(A, ipiv)` — the 1.10 LTS — where the pivot vector costs one `O(n)`
allocation per call. See [`HAS_PREALLOCATED_GETRF`](@ref).

# Example

```jldoctest; setup = :(using SimpleSolvers, Random; using SimpleSolvers: inv; Random.seed!(123))
A = [1. 2. 3.; 5. 7. 11.; 13. 17. 19.]
v = rand(3)
ls = LinearProblem(A, v)

solve(LapackLU(), ls) ≈ inv(A) * v

# output

true
```
"""
struct LapackLU <: DirectMethod end

"""
    LapackLUSolverCache <: LinearSolverCache

The cache for the [`LapackLU`](@ref) solver.

# Keys
- `A`: the working copy of the matrix, which the factorization overwrites in place,
- `ipiv`: the pivot vector LAPACK's `getrf` fills,
- `info`: `getrf`'s status — the index of the first zero pivot, or `0` on success,
- `factorized`: whether [`factorize!`](@ref) has run at all, which `info == 0` cannot express.

The cache holds the *pieces* of the factorization rather than a `LinearAlgebra.LU` object, so
that [`factorize!`](@ref) can be called any number of times without allocating: `LU` is
immutable, so a fresh one would have to be built and boxed into a field on every
refactorization, and its `ipiv` freshly allocated. Both are `O(n)`-to-`O(1)` costs against an
`O(n^3)` factorization, but a nonlinear solve in a time-stepping loop refactorizes on every
step of every step, and `LU` — the method this one sits beside — allocates nothing at all.
(On a Julia without `LAPACK.getrf!(A, ipiv)`, `ipiv` is filled from a per-call temporary
instead; see [`HAS_PREALLOCATED_GETRF`](@ref).)

Use [`factorization`](@ref) to get a `LinearAlgebra.LU` view of these pieces when one is
actually wanted (for `det`, say).

The pivot vector is typed `Vector{LinearAlgebra.BlasInt}` rather than `Vector{Int}`: that is
what `getrf` fills, and the two are not the same type under a 32-bit-integer BLAS.
"""
mutable struct LapackLUSolverCache{T,AT<:AbstractMatrix{T}} <: LinearSolverCache{T}
    A::AT
    ipiv::Vector{LinearAlgebra.BlasInt}
    info::LinearAlgebra.BlasInt
    factorized::Bool
end

function LinearSolverCache(::LapackLU, A::AbstractMatrix{T}) where {T}
    T <: LinearAlgebra.BlasFloat || throw(ArgumentError(
        "LapackLU is restricted to the element types LAPACK provides, i.e. Float32, " *
        "Float64, ComplexF32 and ComplexF64, but got $(T); use LU() instead"))
    n = checksquare(A)
    # a copy, not `undef`, so that the single-argument `factorize!` below has something to
    # factorize — as it does for `LU`, whose cache is likewise seeded from `A`
    Ā = Matrix{T}(A)
    LapackLUSolverCache{T,typeof(Ā)}(Ā, zeros(LinearAlgebra.BlasInt, n), 0, false)
end

"""
    checkfactorized(lsolver::LinearSolver{T,LapackLU})

Throw an `ArgumentError` if [`factorize!`](@ref) has not been called on `lsolver` yet.

The [`LapackLU`](@ref) counterpart of the `perms[1] == 0` guard in [`LU`](@ref)'s
[`ldiv!`](@ref): without it, `getrs` would be handed an all-zero pivot vector and return
garbage rather than complain.
"""
function checkfactorized(lsolver::LinearSolver{T,LapackLU}) where {T}
    cache(lsolver).factorized || throw(ArgumentError(
        "the LapackLU solver has not been factorized yet; call factorize! before ldiv!/solve!."))
    nothing
end

"""
    factorization(lsolver::LinearSolver{T,LapackLU})

A `LinearAlgebra.LU` view of the factorization held in the cache.

This wraps the cache's arrays rather than copying them, so it is only valid until the next
[`factorize!`](@ref). Neither [`ldiv!`](@ref) nor [`singular_index`](@ref) goes through it —
building it allocates, and they are on the hot path — but it is what to reach for when a
`LinearAlgebra.Factorization` is what you want, e.g. `det(factorization(lsolver))`.
"""
function factorization(lsolver::LinearSolver{T,LapackLU}) where {T}
    checkfactorized(lsolver)
    c = cache(lsolver)
    LinearAlgebra.LU(c.A, c.ipiv, c.info)
end

"""
    singular_index(lsolver::LinearSolver{T,LapackLU})

The zero-pivot index LAPACK's `getrf` reported (its `info`), or `0` if the factorization
succeeded.
"""
function singular_index(lsolver::LinearSolver{T,LapackLU}) where {T}
    checkfactorized(lsolver)
    Int(cache(lsolver).info)
end

"""
    HAS_PREALLOCATED_GETRF

Whether this Julia's `LinearAlgebra.LAPACK.getrf!` accepts a pre-allocated pivot vector.

`getrf!(A, ipiv)` arrived after the 1.10 LTS, and it is the whole reason [`factorize!`](@ref)
can be allocation-free. Where it is missing, the one-argument `getrf!(A)` is used and its
pivot vector copied into the cache, which costs one `O(n)` allocation per factorization — the
working matrix is reused either way, and [`ldiv!`](@ref) is allocation-free either way,
because `getrs!` has always taken the pivot vector as an argument.

This is a feature check rather than a `VERSION` comparison, so the exact release it arrived in
does not have to be tracked here.
"""
const HAS_PREALLOCATED_GETRF =
    hasmethod(LinearAlgebra.LAPACK.getrf!, Tuple{Matrix{Float64},Vector{LinearAlgebra.BlasInt}})

@static if HAS_PREALLOCATED_GETRF
    function _getrf!(c::LapackLUSolverCache)
        _, _, c.info = LinearAlgebra.LAPACK.getrf!(c.A, c.ipiv; check=false)
        c
    end
else
    function _getrf!(c::LapackLUSolverCache)
        _, ipiv, info = LinearAlgebra.LAPACK.getrf!(c.A; check=false)
        copyto!(c.ipiv, ipiv)
        c.info = info
        c
    end
end

"""
    factorize!(lsolver::LinearSolver{T,LapackLU}[, A])

Factorize with `LinearAlgebra.lu!`, in place in `cache(lsolver).A`. With two arguments `A` is
first copied into the cache; with one, whatever the cache already holds is factorized.

The factorization is not checked here. A singular matrix is reported when the factorization
is *used*, by [`ldiv!`](@ref), so that a caller that factorizes speculatively — as a
quasi-Newton method does — is not interrupted by a matrix it may never solve with.

!!! warning
    As for [`LU`](@ref), `lu!` overwrites `cache(lsolver).A` with the factors, so the
    single-argument form is good for exactly one call; calling it twice would factorize the
    factors. Use the two-argument form to refactorize.
"""
function factorize!(lsolver::LinearSolver{T,LapackLU}) where {T}
    c = cache(lsolver)
    Base.require_one_based_indexing(c.A)
    # `getrf!` rather than `lu!`: with a pre-allocated `ipiv` where this Julia has that form,
    # and either way without the `LU` object `lu!` would allocate to wrap the result.
    # See `HAS_PREALLOCATED_GETRF` and `LapackLUSolverCache`.
    _getrf!(c)
    c.factorized = true
    lsolver
end

function factorize!(lsolver::LinearSolver{T,LapackLU}, A::AbstractMatrix{T}) where {T}
    c = cache(lsolver)
    axes(A) == axes(c.A) || throw(DimensionMismatch(
        "the matrix to factorize has axes $(axes(A)), but the LapackLU cache was built for " *
        "$(axes(c.A)); allocate a new LinearSolver for a differently sized problem"))
    copyto!(c.A, A)
    factorize!(lsolver)
end

factorize!(lsolver::LinearSolver{T,LapackLU}, ls::LinearProblem{T}) where {T} =
    factorize!(lsolver, matrix(ls))

function LinearAlgebra.ldiv!(x::AbstractVector{T}, lsolver::LinearSolver{T,LapackLU}, b::AbstractVector{T}) where {T}
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
function _getrs!(c::LapackLUSolverCache{T}, x::AbstractVector{T}) where {T}
    if x isa StridedVector{T} && stride(x, 1) == 1
        LinearAlgebra.LAPACK.getrs!('N', c.A, c.ipiv, x)
    else
        y = Vector{T}(x)
        LinearAlgebra.LAPACK.getrs!('N', c.A, c.ipiv, y)
        copyto!(x, y)
    end
    x
end

function solve!(solution::AbstractVector, lsolver::LinearSolver{T,LapackLU}, ls::LinearProblem) where {T}
    factorize!(lsolver, matrix(ls))
    ldiv!(solution, lsolver, rhs(ls))
    solution
end

function solve!(solution::AbstractVector, lsolver::LinearSolver{T,LapackLU}, A::AbstractMatrix, b::AbstractVector) where {T}
    factorize!(lsolver, A)
    ldiv!(solution, lsolver, b)
    solution
end

function solve!(solution::AbstractVector, lsolver::LinearSolver{T,LapackLU}, b::AbstractVector) where {T}
    ldiv!(solution, lsolver, b)
end

function solve!(lsolver::LinearSolver{T,LapackLU}, args...) where {T}
    x = alloc_x(@view cache(lsolver).A[1, :])
    solve!(x, lsolver, args...)
    x
end

"""
    solve(method::LapackLU, ls::LinearProblem)
    solve(method::LapackLU, A, b)

Allocate a [`LinearSolver`](@ref), factorize and solve in one call.

The counterpart of [`solve(::LU, ::LinearProblem)`](@ref). Convenient for a one-off system;
for a solve inside a loop, build the [`LinearSolver`](@ref) once and call
[`factorize!`](@ref) and [`ldiv!`](@ref) on it instead.
"""
function solve(method::LapackLU, ls::LinearProblem)
    lsolver = LinearSolver(method, ls)
    solve!(lsolver, ls)
end

solve(method::LapackLU, A::AbstractMatrix, b::AbstractVector) =
    solve(method, LinearProblem(A, b))
