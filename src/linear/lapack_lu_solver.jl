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
- `A`: the working copy of the matrix, which `lu!` overwrites in place,
- `factorization`: the `LinearAlgebra.LU` object built on it, or `missing` before the first
  call to [`factorize!`](@ref).

The struct is mutable because `factorization` is replaced on every refactorization, while the
storage it points into — `A` — is allocated once and reused.

The pivot vector is typed `Vector{LinearAlgebra.BlasInt}` rather than `Vector{Int}`: that is
what LAPACK's `getrf` fills, and the two are not the same type under a 32-bit-integer BLAS.
"""
mutable struct LapackLUSolverCache{T,AT<:AbstractMatrix{T}} <: LinearSolverCache{T}
    A::AT
    factorization::Union{Missing,LinearAlgebra.LU{T,AT,Vector{LinearAlgebra.BlasInt}}}
end

function LinearSolverCache(::LapackLU, A::AbstractMatrix{T}) where {T}
    T <: LinearAlgebra.BlasFloat || throw(ArgumentError(
        "LapackLU is restricted to the element types LAPACK provides, i.e. Float32, " *
        "Float64, ComplexF32 and ComplexF64, but got $(T); use LU() instead"))
    checksquare(A)
    # a copy, not `undef`, so that the single-argument `factorize!` below has something to
    # factorize — as it does for `LU`, whose cache is likewise seeded from `A`
    Ā = Matrix{T}(A)
    LapackLUSolverCache{T,typeof(Ā)}(Ā, missing)
end

"""
    factorization(lsolver::LinearSolver{T,LapackLU})

The `LinearAlgebra.LU` object stored in the cache, or an `ArgumentError` if
[`factorize!`](@ref) has not been called yet.
"""
function factorization(lsolver::LinearSolver{T,LapackLU}) where {T}
    F = cache(lsolver).factorization
    ismissing(F) && throw(ArgumentError(
        "the LapackLU solver has not been factorized yet; call factorize! before ldiv!/solve!."))
    F
end

"""
    singular_index(lsolver::LinearSolver{T,LapackLU})

The zero-pivot index LAPACK reported for the stored factorization (its `info`), or `0` if the
factorization succeeded.
"""
singular_index(lsolver::LinearSolver{T,LapackLU}) where {T} = Int(factorization(lsolver).info)

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
    c.factorization = LinearAlgebra.lu!(c.A; check=false)
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
    F = factorization(lsolver)
    @assert axes(x, 1) == axes(b, 1) == axes(cache(lsolver).A, 1)
    Base.require_one_based_indexing(x, b)
    LinearAlgebra.issuccess(F) || throw(SingularException(Int(F.info)))
    # `ldiv!(F, x)` solves in place, so the right-hand side has to be moved into `x` first;
    # this is a no-op when the caller passes the same vector for both.
    x === b || copyto!(x, b)
    LinearAlgebra.ldiv!(F, x)
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
