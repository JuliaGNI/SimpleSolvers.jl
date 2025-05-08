"""
    struct LUSolver <: LinearSolver

A custom implementation of an LU solver.

Routines that use `LUSolver` include [`factorize!`](@ref) and [`ldiv!`](@ref).
In practice the `LUSolver` is used by calling its constructor together with [`ldiv!`](@ref) as shown in the *Example section* of this docstring.

# Example

We use the `LUSolver` together with [`ldiv!`](@ref) to compute multiplication of a matrix inverse onto a vector (from the left):

```jldoctest; setup = :(using SimpleSolvers, LinearAlgebra, Random; Random.seed!(123))
A = [1. 2. 3.; 5. 7. 11.; 13. 17. 19.]

lu = LUSolver(A)
v = rand(3)
x = similar(v)

ldiv!(x, lu, v) ≈ inv(A) * v

# output

true
```

When calling `LUSolver` on an integer alone, a matrix with all zeros is allocated:

```jldoctest; setup = :(using SimpleSolvers)
LUSolver{Float32}(2)

# output

LUSolver{Float32}(2, Float32[0.0 0.0; 0.0 0.0], [1, 2], [1, 2], 1)
```

# Keys

- `n::Int`
- `A::Matrix{T}`
- `pivots::Vector{Int}`
- `perms::Vector{Int}`
- `info::Int`
"""
mutable struct LUSolver{T} <: LinearSolver{T}
    n::Int
    A::Matrix{T}
    pivots::Vector{Int}
    perms::Vector{Int}
    info::Int
end

function LUSolver(A::AbstractMatrix{T}) where {T}
    n = checksquare(A)
    lu = LUSolver{T}(n, zero(A), zeros(Int, n), zeros(Int, n), 0)
    factorize!(lu, A)
end

LUSolver{T}(n::Int) where {T} = LUSolver(zeros(T, n, n))

"""
    factorize!(lu, A)

Factorize the matrix `A` and store the result in `lu.A`.

# Examples

```jldoctest; setup = :(using SimpleSolvers)
A = [1. 2. 3.; 5. 7. 11.; 13. 17. 19.]
lu = LUSolver{Float64}(3, similar(A), zeros(Int, 3), zeros(Int, 3), 0)
factorize!(lu, A)
lu.A

# output

3×3 Matrix{Float64}:
 13.0        17.0       19.0
  0.0769231   0.692308   1.53846
  0.384615    0.666667   2.66667
```
Here lu.A stores the factorized result. If we want to save this factorized matrix in the same `A` to save memory we can write:
```jldoctest; setup = :(using SimpleSolvers)
A = [1. 2. 3.; 5. 7. 11.; 13. 17. 19.]
lu = LUSolver{Float64}(3, A, zeros(Int, 3), zeros(Int, 3), 0)
factorize!(lu, A)
A

# output

3×3 Matrix{Float64}:
 13.0        17.0       19.0
  0.0769231   0.692308   1.53846
  0.384615    0.666667   2.66667
```
"""
function factorize!(lu::LUSolver{T}, A::AbstractMatrix{T}; pivot=true) where {T}
    copy!(lu.A, A)
    
    @inbounds for i in eachindex(lu.perms)
        lu.perms[i] = i
    end

    @inbounds for k ∈ 1:lu.n
        # find index max
        kp = pivot ? find_maximum_value(lu.A[:, k], k) : k
        
        lu.pivots[k] = kp
        lu.perms[k], lu.perms[kp] = lu.perms[kp], lu.perms[k]

        if lu.A[kp,k] != 0
            if k != kp
                # Interchange
                for i in 1:lu.n
                    tmp = lu.A[k,i]
                    lu.A[k,i] = lu.A[kp, i]
                    lu.A[kp,i] = tmp
                end
            end
            # Scale first column
            Akkinv = inv(lu.A[k,k])
            for i in k+1:lu.n
                lu.A[i,k] *= Akkinv
            end
        elseif lu.info == 0
            lu.info = k
        end
        # Update the rest
        for j in k+1:lu.n
            for i in k+1:lu.n
                lu.A[i,j] -= lu.A[i,k] * lu.A[k,j]
            end
        end
    end

    lu
end

"""
    find_maximum_value(v, k)

Find the maximum value of vector `v` starting from the index `k`.
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

Compute `inv(lu.A) * b` by utilizing the factorization in the [`LUSolver`](@ref) and store the result in `x`.
"""
function LinearAlgebra.ldiv!(x::AbstractVector{T}, lu::LUSolver{T}, b::AbstractVector{T}) where {T}
    @assert axes(x,1) == axes(b,1) == axes(lu.A,1) == axes(lu.A,2)

    @inbounds for i in 1:lu.n
        x[i] = b[lu.perms[i]]
    end

    @inbounds for i in 2:lu.n
        s = zero(T)
        for j in 1:i-1
            s += lu.A[i,j] * x[j]
        end
        x[i] -= s
    end

    x[lu.n] /= lu.A[lu.n,lu.n]
    @inbounds for i in lu.n-1:-1:1
        s = zero(T)
        for j in i+1:lu.n
            s += lu.A[i,j] * x[j]
        end
        x[i] -= s
        x[i] /= lu.A[i,i]
    end

    x
end
