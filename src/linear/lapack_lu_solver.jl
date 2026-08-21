"""
    struct LapackLU <: DirectMethod

A LAPACK-backed LU solver, meant to solve a [`LinearProblem`](@ref).

Where [`LU`](@ref) is a self-contained implementation that works for any number type and for
static matrices, this method delegates the factorization to `LinearAlgebra.lu!` and therefore
to LAPACK. It is restricted to the element types LAPACK handles — `Float32`, `Float64`,
`ComplexF32` and `ComplexF64` — and to plain `Matrix` storage.

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

const LapackEltype = Union{Float32,Float64,ComplexF32,ComplexF64}

"""
    LapackLUSolverCache <: LinearSolverCache

The cache for the [`LapackLU`](@ref) solver.

# Keys
- `A`: the working copy of the matrix, which `lu!` overwrites in place,
- `factorization`: the `LinearAlgebra.LU` object built on it, or `missing` before the first
  call to [`factorize!`](@ref).

The field is mutable because the factorization object is replaced on every refactorization
while the storage it points into is reused.
"""
mutable struct LapackLUSolverCache{T,AT<:AbstractMatrix{T}} <: LinearSolverCache{T}
    A::AT
    factorization::Union{Missing,LinearAlgebra.LU{T,AT,Vector{Int}}}
end

function LinearSolverCache(::LapackLU, A::AbstractMatrix{T}) where {T}
    T <: LapackEltype || throw(ArgumentError(
        "LapackLU is restricted to the element types LAPACK provides, i.e. Float32, " *
        "Float64, ComplexF32 and ComplexF64, but got $(T); use LU() instead"))
    n = checksquare(A)
    Ā = Matrix{T}(undef, n, n)
    LapackLUSolverCache{T,typeof(Ā)}(Ā, missing)
end

"""
    factorize!(lsolver::LinearSolver{T,LapackLU}, A)

Copy `A` into the cache and factorize it with `LinearAlgebra.lu!`.

The factorization is not checked here. A singular matrix is reported when the factorization
is *used*, by [`ldiv!`](@ref), so that a caller that factorizes speculatively — as a
quasi-Newton method does — is not interrupted by a matrix it may never solve with.
"""
function factorize!(lsolver::LinearSolver{T,LapackLU}, A::AbstractMatrix) where {T}
    c = cache(lsolver)
    copyto!(c.A, A)
    c.factorization = LinearAlgebra.lu!(c.A; check=false)
    lsolver
end

factorize!(lsolver::LinearSolver{T,LapackLU}, ls::LinearProblem{T}) where {T} =
    factorize!(lsolver, ls.A)

function LinearAlgebra.ldiv!(x::AbstractVector{T}, lsolver::LinearSolver{T,LapackLU}, b::AbstractVector{T}) where {T}
    c = cache(lsolver)
    ismissing(c.factorization) && throw(ArgumentError(
        "the LapackLU solver has not been factorized yet; call factorize! first"))
    LinearAlgebra.issuccess(c.factorization) || throw(LinearAlgebra.SingularException(0))
    copyto!(x, b)
    LinearAlgebra.ldiv!(c.factorization, x)
    x
end

function solve!(solution::AbstractVector, lsolver::LinearSolver{T,LapackLU}, ls::LinearProblem) where {T}
    factorize!(lsolver, ls.A)
    ldiv!(solution, lsolver, rhs(ls))
    solution
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
    solve!(zero(rhs(ls)), lsolver, ls)
end

solve(method::LapackLU, A::AbstractMatrix, b::AbstractVector) =
    solve(method, LinearProblem(A, b))
