"""
    struct LU <: LinearSolverMethod

A custom implementation of an LU solver, meant to solve a [`LinearSystem`](@ref).

Routines that use the LU solver include [`factorize!`](@ref), [`ldiv!`](@ref) and [`solve!`](@ref).
In practice the `LU` solver is used by calling the [`LinearSolver`](@ref) constructor and [`ldiv!`](@ref) or [`solve!`](@ref), or with an instance of `LU` as an argument directly, as shown in the *Example section* of this docstring.

# constructor

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

Note that if we do not supply an explicit keyword `static`, the corresponding field is `missing` (as in the first case). Also see [`_static`](@ref).

# Example

We use the `LU` together with [`solve`](@ref) to solve a linear system:

```jldoctest; setup = :(using SimpleSolvers, Random; using SimpleSolvers: inv, update!; Random.seed!(123))
A = [1. 2. 3.; 5. 7. 11.; 13. 17. 19.]
v = rand(3)
ls = LinearSystem(A, v)
update!(ls, A, v)

lu = LU()

solve(lu, ls) ≈ inv(A) * v

# output

true
```
"""
struct LU{ST <: Union{Missing, Bool}} <: LinearSolverMethod 
    static::ST
    pivot::Bool

    LU(; pivot=true, static=missing) = new{typeof(static)}(static, pivot)
end

"""
Threshold for the maximum size a static matrix should have.
"""
const N_STATIC_THRESHOLD = 10

"""
    _static(A)

Determine whether to allocate a `StaticArray` or simply copy the input array.
This is used when calling [`LinearSolverCache`](@ref) on [`LU`](@ref).
Every matrix that is smaller or equal to [`N_STATIC_THRESHOLD`](@ref) is turned into a `StaticArray` as a consequence.
"""
_static(A::AbstractMatrix)::Bool = length(axes(A, 1)) ≤ N_STATIC_THRESHOLD ? true : false

"""
    LUSolverCache <: LinearSolverCache

# Keys
- `A`: the factorized matrix `A`,
- `pivots`:
- `perms`:
- `info`
"""
mutable struct LUSolverCache{T, AT <: AbstractMatrix{T}} <: LinearSolverCache{T}
    A::AT
    pivots::Vector{Int}
    perms::Vector{Int}
    info::Int
end

function LinearSolverCache(::LU{Missing}, A::AbstractMatrix{T}) where {T}
    n = checksquare(A)
    Ā = _static(A) ? MMatrix{size(A)...}(copy(A)) : copy(A)
    LUSolverCache{T, typeof(Ā)}(Ā, zeros(Int, n), zeros(Int, n), 0)
end

function LinearSolverCache(lu::LU{Bool}, A::AbstractMatrix{T}) where {T}
    n = checksquare(A)
    Ā = lu.static ? MMatrix{size(A)...}(copy(A)) : copy(A)
    LUSolverCache{T, typeof(Ā)}(Ā, zeros(Int, n), zeros(Int, n), 0)
end

function solve!(solution::AbstractVector, lsolver::LinearSolver{T, LUT}, ls::LinearSystem) where {T, LUT <: LU}
    cache(lsolver).A .= ls.A
    factorize!(lsolver)
    ldiv!(solution, lsolver, rhs(ls))
    solution
end

function solve!(solution::AbstractVector, lsolver::LinearSolver{T, LUT}, b::AbstractVector) where {T, LUT <: LU}
    ldiv!(solution, lsolver, b)
    solution
end

function solve!(solution::AbstractVector, lsolver::LinearSolver{T, LUT}, A::AbstractMatrix, b::AbstractVector) where {T, LUT <: LU}
    cache(lsolver).A .= A 
    factorize!(lsolver)
    ldiv!(solution, lsolver, b)
    solution
end

function solve!(solution::AbstractVector, lsolver::LinearSolver{T, LUT}, A::AbstractMatrix) where {T, LUT <: LU}
    solve!(solution, lsolver, A, rhs(cache(lsolver)))
end

function solve!(lsolver::LinearSolver{T, LUT}, args...)  where {T, LUT <: LU}
    x = alloc_x(cache(lsolver).A[1, :])
    solve!(x, lsolver, args...)
    x
end

function solve(lu::LU, ls::LinearSystem) 
    lsolver = LinearSolver(lu, ls)
    solve!(lsolver, ls)
end

function solve(lu::LU, A::AbstractMatrix, b::AbstractVector)
    ls = LinearSystem(A, b)
    update!(ls, A, b)
    solve(lu, ls)
end

"""
    factorize!(lsolver::LinearSolver, A)

Factorize the matrix `A` and store the result in `cache(lsolver).A`.
Note that calling `cache` on `lsolver` returns the instance of [`LUSolverCache`](@ref) stored in `lsolver`.

# Examples

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: cache)
A = [1. 2. 3.; 5. 7. 11.; 13. 17. 19.]
y = [1., 0., 0.]
x = similar(y)

lsolver = LinearSolver(LU(; static=false), x)
factorize!(lsolver, A)
cache(lsolver).A

# output

3×3 Matrix{Float64}:
 13.0        17.0       19.0
  0.0769231   0.692308   1.53846
  0.384615    0.666667   2.66667
```
Here `cache(lsolver).A` stores the factorized matrix. If we call `factorize!` with two input arguments as above, the method first copies the matrix `A` into the [`LUSolverCache`](@ref). We can equivalently also do:

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: cache)
A = [1. 2. 3.; 5. 7. 11.; 13. 17. 19.]
y = [1., 0., 0.]

lsolver = LinearSolver(LU(), A)
factorize!(lsolver)
cache(lsolver).A

# output

3×3 StaticArraysCore.MMatrix{3, 3, Float64, 9} with indices SOneTo(3)×SOneTo(3):
 13.0        17.0       19.0
  0.0769231   0.692308   1.53846
  0.384615    0.666667   2.66667
```

Also note the difference between the output types of the two refactorized matrices. This is because we set the keyword `static` to false when calling [`LU`](@ref). Also see [`_static`](@ref).
"""
function factorize!(lsolver::LinearSolver{T, LUT}) where {T, LUT <: LU}
    @inbounds for i in eachindex(cache(lsolver).perms)
        cache(lsolver).perms[i] = i
    end

    n = size(cache(lsolver).A, 1)

    @inbounds for k ∈ axes(cache(lsolver).A, 1)
        kp = method(lsolver).pivot ? find_maximum_value(cache(lsolver).A[:, k], k) : k
        
        cache(lsolver).pivots[k] = kp
        cache(lsolver).perms[k], cache(lsolver).perms[kp] = cache(lsolver).perms[kp], cache(lsolver).perms[k]

        if cache(lsolver).A[kp,k] != 0
            if k != kp
                # Interchange
                for i in 1:n
                    tmp = cache(lsolver).A[k,i]
                    cache(lsolver).A[k,i] = cache(lsolver).A[kp, i]
                    cache(lsolver).A[kp,i] = tmp
                end
            end
            # Scale first column
            Akkinv = inv(cache(lsolver).A[k,k])
            for i in k+1:n
                cache(lsolver).A[i,k] *= Akkinv
            end
        elseif lu.info == 0
            lu.info = k
        end
        # Update the rest
        for j in k+1:n
            for i in k+1:n
                cache(lsolver).A[i,j] -= cache(lsolver).A[i,k] * cache(lsolver).A[k,j]
            end
        end
    end

    lsolver
end

function factorize!(lsolver::LinearSolver{T, LUT}, A::AbstractMatrix{T}) where {T, LUT <: LU}
    copyto!(cache(lsolver).A, A)
    
    factorize!(lsolver)
end

factorize!(lsolver::LinearSolver{T, LUT}, ls::LinearSystem{T}) where {T, LUT <: LU} = factorize!(lsolver, ls.A)

"""
    find_maximum_value(v, k)

Find the maximum value of vector `v` starting from the index `k`. 
This is used for *pivoting* in [`factorize!`](@ref).
"""
function find_maximum_value(v::AbstractVector{T}, k::Integer) where {T <: Number}
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
    ldiv!(x, lu, b)

Compute `inv(cache(lsolver).A) * b` by utilizing the factorization of the lu solver (see [`LU`](@ref) and [`LinearSolver`](@ref)) and store the result in `x`.
"""
function LinearAlgebra.ldiv!(x::AbstractVector{T}, lsolver::LinearSolver{T, LUT}, b::AbstractVector{T}) where {T, LUT <: LU}
    @assert axes(x,1) == axes(b,1) == axes(cache(lsolver).A,1) == axes(cache(lsolver).A,2)

    n = size(cache(lsolver).A, 1)

    @inbounds for i in 1:n
        x[i] = b[cache(lsolver).perms[i]]
    end

    @inbounds for i in 2:n
        s = zero(T)
        for j in 1:i-1
            s += cache(lsolver).A[i,j] * x[j]
        end
        x[i] -= s
    end

    x[n] /= cache(lsolver).A[n,n]
    @inbounds for i in n-1:-1:1
        s = zero(T)
        for j in i+1:n
            s += cache(lsolver).A[i,j] * x[j]
        end
        x[i] -= s
        x[i] /= cache(lsolver).A[i,i]
    end

    x
end