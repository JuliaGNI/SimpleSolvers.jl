"""
    struct LU <: DirectMethod

A custom implementation of an LU solver, meant to solve a [`LinearProblem`](@ref).

Routines that use the LU solver include [`factorize!`](@ref), [`ldiv!`](@ref) and [`solve!`](@ref).

# Constructor

The constructor is called with either no argument:

```jldoctest; setup = :(using SimpleSolvers)
LU()

# output

LU{Missing}(missing, true)
```

or with `pivot` and `static` as optional booleans:

```jldoctest; setup = :(using SimpleSolvers)
LU(; pivot=true, static=true)

# output

LU{Bool}(true, true)
```

Note that if we do not supply an explicit keyword `static`, the corresponding field is `missing` (as in the first case). In that default case the cache matrix type is chosen by size via [`_static`](@ref): a matrix whose leading dimension does not exceed [`N_STATIC_THRESHOLD`](@ref) yields a mutable static (`MMatrix`) cache, a larger one yields a plain `Matrix`. An explicit `static=true`/`false` forces the choice regardless of the matrix size.

# Example

We use the `LU` together with [`solve`](@ref) to solve a linear system:

```jldoctest; setup = :(using SimpleSolvers, Random; using SimpleSolvers: inv; Random.seed!(123))
A = [1. 2. 3.; 5. 7. 11.; 13. 17. 19.]
v = rand(3)
ls = LinearProblem(A, v)

lu = LU()

solve(lu, ls) ≈ inv(A) * v

# output

true
```

Note that role of [`LinearProblem`](@ref) here.
"""
struct LU{ST<:Union{Missing,Bool}} <: DirectMethod
    static::ST
    pivot::Bool

    LU(; pivot=true, static=missing) = new{typeof(static)}(static, pivot)
end

"""
    LUSolverCache <: LinearSolverCache

The cache for the [`LU`](@ref) solver.

# Keys
- `A`: the factorized matrix `A`,
- `pivots`: a vector of pivots used during factorization,
- `perms`: a vector of permutations used during factorization,
- `info`: stores an index regarding pivoting.
"""
mutable struct LUSolverCache{T,AT<:AbstractMatrix{T}} <: LinearSolverCache{T}
    A::AT
    pivots::Vector{Int}
    perms::Vector{Int}
    info::Int
end

"""
    lucache_eltype(T)

The element type used by the [`LUSolverCache`](@ref) for an input matrix of
element type `T`.  Integer (and other non-fractional) inputs are promoted to a
type that supports division (mirroring `LinearAlgebra.lutype`), so that e.g.
`LinearSolver(LU(), [1 2; 3 4])` works instead of failing in `factorize!`.
"""
lucache_eltype(::Type{T}) where {T} = typeof(oneunit(T) / oneunit(T))

"""
Threshold for the maximum size a static matrix should have. See [`_static`](@ref).
"""
const N_STATIC_THRESHOLD = 10

"""
    _static(A)

Determine whether the [`LUSolverCache`](@ref) for a default [`LU`](@ref) should store
`A` as a mutable static matrix (`MMatrix`) or as a plain `Matrix`.  Every matrix whose
leading dimension is smaller than or equal to [`N_STATIC_THRESHOLD`](@ref) is stored as
an `MMatrix`.

This is only consulted for the default `LU()` (i.e. `LU{Missing}`); an explicit
`static=true`/`false` keyword overrides it.  See the examples in [`factorize!`](@ref).
"""
_static(A::AbstractMatrix)::Bool = length(axes(A, 1)) ≤ N_STATIC_THRESHOLD

function LinearSolverCache(::LU{Missing}, A::AbstractMatrix{T}) where {T}
    n = checksquare(A)
    Tf = lucache_eltype(T)
    Ā = _static(A) ? MMatrix{size(A)...}(Tf.(A)) : Tf.(A)
    LUSolverCache{Tf,typeof(Ā)}(Ā, zeros(Int, n), zeros(Int, n), 0)
end

function LinearSolverCache(lu::LU{Bool}, A::AbstractMatrix{T}) where {T}
    n = checksquare(A)
    Tf = lucache_eltype(T)
    Ā = lu.static ? MMatrix{size(A)...}(Tf.(A)) : Tf.(A)
    LUSolverCache{Tf,typeof(Ā)}(Ā, zeros(Int, n), zeros(Int, n), 0)
end

function solve!(solution::AbstractVector, lsolver::LinearSolver{T,LUT}, ls::LinearProblem) where {T,LUT<:LU}
    cache(lsolver).A .= ls.A
    factorize!(lsolver)
    ldiv!(solution, lsolver, rhs(ls))
    solution
end

function solve!(solution::AbstractVector, lsolver::LinearSolver{T,LUT}, A::AbstractMatrix, b::AbstractVector) where {T,LUT<:LU}
    factorize!(lsolver, A)
    ldiv!(solution, lsolver, b)
    solution
end

function solve!(solution::AbstractVector, lsolver::LinearSolver{T,LUT}, b::AbstractVector) where {T,LUT<:LU}
    ldiv!(solution, lsolver, b)
end

function solve!(lsolver::LinearSolver{T,LUT}, args...) where {T,LUT<:LU}
    x = alloc_x(@view cache(lsolver).A[1, :])
    solve!(x, lsolver, args...)
    x
end

function solve(lu::LU, ls::LinearProblem)
    lsolver = LinearSolver(lu, ls)
    solve!(lsolver, ls)
end

"""
    solve(lu, A, b)

Solve the *linear problem* determined by `A` and `b`.

This is the most straightforward way to solve this system.

# Examples

```jldoctest; setup = :(using SimpleSolvers)
julia> A = [1.; 0.; 0.;; 0.; 2.; 0.;; 0.; 0.; 4.]
3×3 Matrix{Float64}:
 1.0  0.0  0.0
 0.0  2.0  0.0
 0.0  0.0  4.0

julia> b = ones(3)
3-element Vector{Float64}:
 1.0
 1.0
 1.0

julia> solve(LU(), A, b)
3-element StaticArraysCore.SizedVector{3, Float64, Vector{Float64}} with indices SOneTo(3):
 1.0
 0.5
 0.25
```
Compare this to [`solve!(::AbstractVector, ::LinearSolver, ::LinearProblem)`](@ref).
"""
function solve(lu::LU, A::AbstractMatrix, b::AbstractVector)
    solve(lu, LinearProblem(A, b))
end

"""
    factorize!(lsolver::LinearSolver, A)

Factorize the matrix `A` and store the result in `cache(lsolver).A`.

Note that calling [`cache`](@ref) on `lsolver` returns the instance of [`LUSolverCache`](@ref) stored in `lsolver`.

# Examples

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: cache, ldiv!)
julia> A = [1. 2. 3.; 5. 7. 11.; 13. 17. 19.]
3×3 Matrix{Float64}:
  1.0   2.0   3.0
  5.0   7.0  11.0
 13.0  17.0  19.0

julia> x = zeros(3);

julia> lsolver = LinearSolver(LU(; static=false), x);

julia> factorize!(lsolver, A).cache.A
3×3 Matrix{Float64}:
 13.0        17.0       19.0
  0.0769231   0.692308   1.53846
  0.384615    0.666667   2.66667

julia> y = A * ldiv!(x, lsolver, ones(3));

julia> round.(y; digits = 10)
3-element Vector{Float64}:
 1.0
 1.0
 1.0
```
Here `cache(lsolver).A` stores the factorized matrix. If we call `factorize!` with two input arguments as above, the method first copies the matrix `A` into the [`LUSolverCache`](@ref). We can equivalently also do:

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: cache; A = [1. 2. 3.; 5. 7. 11.; 13. 17. 19.])
julia> lsolver = LinearSolver(LU(), A);

julia> factorize!(lsolver).cache.A
3×3 StaticArraysCore.MMatrix{3, 3, Float64, 9} with indices SOneTo(3)×SOneTo(3):
 13.0        17.0       19.0
  0.0769231   0.692308   1.53846
  0.384615    0.666667   2.66667
```

Note the difference between the output types of the two refactorized matrices: the default `LU()` chose a mutable static (`MMatrix`) cache because the matrix is small (see [`_static`](@ref) and [`N_STATIC_THRESHOLD`](@ref)), whereas `LU(; static=false)` forced a plain `Matrix`.

Also see [`ldiv!`](@ref) for how the refactorized matrix is used.
"""
function factorize!(lsolver::LinearSolver{T,LUT}) where {T,LUT<:LU}
    Base.require_one_based_indexing(cache(lsolver).A)

    cache(lsolver).info = 0

    @inbounds for i in eachindex(cache(lsolver).perms)
        cache(lsolver).perms[i] = i
    end

    n = size(cache(lsolver).A, 1)

    @inbounds for k ∈ axes(cache(lsolver).A, 1)
        kp = method(lsolver).pivot ? pivot_index(@view(cache(lsolver).A[:, k]), k) : k

        cache(lsolver).pivots[k] = kp
        cache(lsolver).perms[k], cache(lsolver).perms[kp] = cache(lsolver).perms[kp], cache(lsolver).perms[k]

        if cache(lsolver).A[kp, k] != 0
            if k != kp
                # Interchange
                for i in 1:n
                    tmp = cache(lsolver).A[k, i]
                    cache(lsolver).A[k, i] = cache(lsolver).A[kp, i]
                    cache(lsolver).A[kp, i] = tmp
                end
            end
            # Scale first column
            Akkinv = inv(cache(lsolver).A[k, k])
            for i in k+1:n
                cache(lsolver).A[i, k] *= Akkinv
            end
        elseif cache(lsolver).info == 0
            cache(lsolver).info = k
        end
        # Update the rest
        for j in k+1:n
            for i in k+1:n
                cache(lsolver).A[i, j] -= cache(lsolver).A[i, k] * cache(lsolver).A[k, j]
            end
        end
    end

    lsolver
end

function factorize!(lsolver::LinearSolver{T,LUT}, A::AbstractMatrix{T}) where {T,LUT<:LU}
    copyto!(cache(lsolver).A, A)

    factorize!(lsolver)
end

factorize!(lsolver::LinearSolver{T,LUT}, ls::LinearProblem{T}) where {T,LUT<:LU} = factorize!(lsolver, ls.A)

"""
    pivot_index(v, k)

Return the index (starting from `k`) of the entry of `v` with the largest absolute
value.

This is used for *pivoting* in [`factorize!`](@ref).
"""
function pivot_index(v::AbstractVector{T}, k::Integer) where {T<:Number}
    Base.require_one_based_indexing(v)
    kp = k
    amax = real(zero(T))
    for i in k:length(v)
        absi = abs(v[i])
        if absi > amax
            kp = i
            amax = absi
        end
    end
    kp
end

"""
    ldiv!(x, lsolver, b)

Compute `inv(cache(lsolver).A) * b` by utilizing the factorization of the lu solver (see [`LU`](@ref) and [`LinearSolver`](@ref)) and store the result in `x`.

# Examples

```jldoctest; setup = :(using SimpleSolvers; using LinearAlgebra:ldiv!)
julia> A = [1.; 0.; 0.;; 0.; 2.; 0.;; 0.; 0.; 4.]
3×3 Matrix{Float64}:
 1.0  0.0  0.0
 0.0  2.0  0.0
 0.0  0.0  4.0

julia> b = [1., 1., 1.]
3-element Vector{Float64}:
 1.0
 1.0
 1.0

julia> s = LinearSolver(LU(), A); factorize!(s); x = zeros(3)
3-element Vector{Float64}:
 0.0
 0.0
 0.0

julia> ldiv!(x, s, b)
3-element Vector{Float64}:
 1.0
 0.5
 0.25

```

!!! info
    Note that we need to call [`factorize!`](@ref) here after having allocated the [`LinearSolver`](@ref).
"""
function LinearAlgebra.ldiv!(x::AbstractVector{T}, lsolver::LinearSolver{T,LUT}, b::AbstractVector{T}) where {T,LUT<:LU}
    @assert axes(x, 1) == axes(b, 1) == axes(cache(lsolver).A, 1) == axes(cache(lsolver).A, 2)

    Base.require_one_based_indexing(x, b, cache(lsolver).A)

    cache(lsolver).info == 0 || throw(SingularException(cache(lsolver).info))

    n = size(cache(lsolver).A, 1)

    # the permutation gather below corrupts the result if `x` and `b` alias
    if x === b
        b = copy(b)
    end

    @inbounds for i in 1:n
        x[i] = b[cache(lsolver).perms[i]]
    end

    @inbounds for i in 2:n
        s = zero(T)
        for j in 1:i-1
            s += cache(lsolver).A[i, j] * x[j]
        end
        x[i] -= s
    end

    x[n] /= cache(lsolver).A[n, n]
    @inbounds for i in n-1:-1:1
        s = zero(T)
        for j in i+1:n
            s += cache(lsolver).A[i, j] * x[j]
        end
        x[i] -= s
        x[i] /= cache(lsolver).A[i, i]
    end

    x
end
