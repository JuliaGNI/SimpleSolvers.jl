"""
    struct LUSolver <: LinearSolver

A custom implementation of an LU solver, meant to solve a [`LinearSystem`](@ref).

Routines that use `LUSolver` include [`factorize!`](@ref), [`ldiv!`](@ref) and [`solve!`](@ref).
In practice the `LUSolver` is used by calling its constructor together with [`ldiv!`](@ref) or [`solve!`](@ref), as shown in the *Example section* of this docstring.

# Example

We use the `LUSolver` together with [`ldiv!`](@ref) to compute multiplication of a matrix inverse onto a vector (from the left):

```jldoctest; setup = :(using SimpleSolvers, LinearAlgebra, Random; Random.seed!(123))
A = [1. 2. 3.; 5. 7. 11.; 13. 17. 19.]
v = rand(3)
ls = LinearSystem(copy(A), v)

lu = LUSolver(ls)

solution(ls) ≈ inv(A) * v

# output

true
```

When calling `LUSolver` on a matrix together with a vector the constructor automatically calls [`solve!`](@ref), i.e. solves the system. When calling `LUSolver` on an integer alone, a matrix with all zeros is allocated:

```jldoctest; setup = :(using SimpleSolvers)
LUSolver{Float32}(2)

# output

LUSolver{Float32, LinearSystem{Float32, Vector{Float32}, Matrix{Float32}}}(2, LinearSystem{Float32, Vector{Float32}, Matrix{Float32}}(Float32[NaN, NaN], Float32[NaN NaN; NaN NaN], Float32[NaN, NaN]), [1, 2], [1, 2], 0, true)
```

# Keys

- `n::Int`
- `A::Matrix{T}`
- `pivots::Vector{Int}`
- `perms::Vector{Int}`
- `info::Int`
- `pivot::Bool`
"""
mutable struct LUSolver{T, LST <: LinearSystem{T}} <: LinearSolver{T}
    n::Int
    linearsystem::LST
    pivots::Vector{Int}
    perms::Vector{Int}
    info::Int
    pivot::Bool
end

"""
    linearsystem(lu)

Access the [`LinearSystem`](@ref) stored in the [`LUSolver`](@ref).
"""
linearsystem(lu::LUSolver) = lu.linearsystem

function LUSolver(ls::LST; pivot = true) where {T, LST <: LinearSystem{T}}
    n = checksquare(Matrix(ls))
    lu = LUSolver{T, LST}(n, ls, zeros(Int, n), zeros(Int, n), 0, pivot)
    solve!(lu)
end

function LUSolver{T}(n::Int; kwargs...) where {T}
    ls = LinearSystem{T}(n)
    LUSolver(ls; kwargs...)
end

"""
    factorize!(lu, A)

Factorize the matrix `A` and store the result in `Matrix(linearsystem(lu))`.

# Examples

```jldoctest; setup = :(using SimpleSolvers)
A = [1. 2. 3.; 5. 7. 11.; 13. 17. 19.]
y = [1., 0., 0.]

ls = LinearSystem(similar(A), y)
lu = LUSolver{Float64, typeof(ls)}(3, ls, zeros(Int, 3), zeros(Int, 3), 0, true)
factorize!(lu, A)
Matrix(linearsystem(lu))

# output

3×3 Matrix{Float64}:
 13.0        17.0       19.0
  0.0769231   0.692308   1.53846
  0.384615    0.666667   2.66667
```
Here `Matrix(linearsystem(lu))` stores the factorized result. If we want to save this factorized matrix in the same `A` to save memory we can write:
```jldoctest; setup = :(using SimpleSolvers)
A = [1. 2. 3.; 5. 7. 11.; 13. 17. 19.]
x = [1., 2., 3.]
y = [2., 3., 4.]
ls = LinearSystem(x, one(A), y)
lu = LUSolver(ls)
factorize!(lu, A)
Matrix(ls)

# output

3×3 Matrix{Float64}:
 13.0        17.0       19.0
  0.0769231   0.692308   1.53846
  0.384615    0.666667   2.66667
```
"""
function factorize!(lu::LUSolver{T}, A::AbstractMatrix{T}; pivot=lu.pivot) where {T}
    copyto!(Matrix(linearsystem(lu)), A)
    
    @inbounds for i in eachindex(lu.perms)
        lu.perms[i] = i
    end

    @inbounds for k ∈ 1:lu.n
        kp = pivot ? find_maximum_value(Matrix(linearsystem(lu))[:, k], k) : k
        
        lu.pivots[k] = kp
        lu.perms[k], lu.perms[kp] = lu.perms[kp], lu.perms[k]

        if Matrix(linearsystem(lu))[kp,k] != 0
            if k != kp
                # Interchange
                for i in 1:lu.n
                    tmp = Matrix(linearsystem(lu))[k,i]
                    Matrix(linearsystem(lu))[k,i] = Matrix(linearsystem(lu))[kp, i]
                    Matrix(linearsystem(lu))[kp,i] = tmp
                end
            end
            # Scale first column
            Akkinv = inv(Matrix(linearsystem(lu))[k,k])
            for i in k+1:lu.n
                Matrix(linearsystem(lu))[i,k] *= Akkinv
            end
        elseif lu.info == 0
            lu.info = k
        end
        # Update the rest
        for j in k+1:lu.n
            for i in k+1:lu.n
                Matrix(linearsystem(lu))[i,j] -= Matrix(linearsystem(lu))[i,k] * Matrix(linearsystem(lu))[k,j]
            end
        end
    end

    lu
end

"""
    factorize!(lu::LUSolver)

Factorize the matrix stored in [`linearsystem`](@ref)`(lu)` via an LU decomposition.
"""
factorize!(lu::LUSolver; kwargs...) = factorize!(lu, Matrix(linearsystem(lu)); kwargs...)

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

Compute `inv(Matrix(linearsystem(lu))) * b` by utilizing the factorization in the [`LUSolver`](@ref) and store the result in `x`.
"""
function LinearAlgebra.ldiv!(x::AbstractVector{T}, lu::LUSolver{T}, b::AbstractVector{T}) where {T}
    @assert axes(x,1) == axes(b,1) == axes(Matrix(linearsystem(lu)),1) == axes(Matrix(linearsystem(lu)),2)

    @inbounds for i in 1:lu.n
        x[i] = b[lu.perms[i]]
    end

    @inbounds for i in 2:lu.n
        s = zero(T)
        for j in 1:i-1
            s += Matrix(linearsystem(lu))[i,j] * x[j]
        end
        x[i] -= s
    end

    x[lu.n] /= Matrix(linearsystem(lu))[lu.n,lu.n]
    @inbounds for i in lu.n-1:-1:1
        s = zero(T)
        for j in i+1:lu.n
            s += Matrix(linearsystem(lu))[i,j] * x[j]
        end
        x[i] -= s
        x[i] /= Matrix(linearsystem(lu))[i,i]
    end

    x
end

@doc raw"""
    solution(lu::LUSolver)

Get the solution (the ``x``) contained in [`linearsystem`](@ref)`(lu)`.
"""
solution(lu::LUSolver) = solution(linearsystem(lu))

"""
    solve!(lu::LUSolver)

Solve the [`LinearSystem`](@ref) stored in `lu` and store the result in [`solution`](@ref)`(lu)`.
"""
function solve!(lu::LUSolver)
    factorize!(lu)
    ldiv!(solution(lu), lu, rhs(linearsystem(lu)))
    lu.solved = true
    lu
end