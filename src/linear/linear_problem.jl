"""
Encompasses the [`NoLinearProblem`](@ref) and the [`LinearProblem`](@ref). Subtyped from `AbstractProblem`, coming from `GeometricBase`.
"""
abstract type AbstractLinearProblem <: AbstractProblem end

"""
A *dummy linear system* used for the *fixed point iterator* ([`Picard`](@ref)).
"""
struct NoLinearProblem <: AbstractLinearProblem end

"""
    LinearProblem

A `LinearProblem` describes ``Ax = y``, where we want to solve for ``x``.

# Keys
- `A`
- `y`

# Constructors

A `LinearProblem` can be allocated by calling:

```julia
LinearProblem(A, y)
LinearProblem(A)
LinearProblem(y)
LinearProblem{T}(n, m)
LinearProblem{T}(n)
```

`LinearProblem(A, y)` stores *copies* of `A` and `y`, so the problem is ready to
solve right after construction (and later mutations of the caller's arrays do not
affect the stored copies):

```jldoctest; setup = :(using SimpleSolvers)
A = [1. 2. 3.; 4. 5. 6.; 7. 8. 9.]
y = [1., 2., 3.]
ls = LinearProblem(A, y)

# output

LinearProblem{Float64, Vector{Float64}, Matrix{Float64}}([1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0], [1.0, 2.0, 3.0])
```

The size-only constructors (`LinearProblem(A)`, `LinearProblem(y)`,
`LinearProblem{T}(n[, m])`) allocate the unspecified parts as `NaN`s; use
[`update!`](@ref) to fill the system with values:

```jldoctest; setup = :(using SimpleSolvers; A = [1. 2. 3.; 4. 5. 6.; 7. 8. 9.]; y = [1., 2., 3.])
ls = LinearProblem(y)
update!(ls, A, y)

# output

LinearProblem{Float64, Vector{Float64}, Matrix{Float64}}([1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0], [1.0, 2.0, 3.0])
```
"""
mutable struct LinearProblem{T, VT <: AbstractVector{T}, AT <: AbstractMatrix{T}} <:
               AbstractLinearProblem
    A::AT
    y::VT
    function LinearProblem(A::AT, y::VT) where {
            T <: Number, VT <: AbstractVector{T}, AT <: AbstractMatrix{T}}
        @assert length(y) == size(A, 1)
        new{T, VT, AT}(copy(A), copy(y))
    end
end

# `alloc_rhs` rather than `alloc_x(A[:, 1])`: a column of a sparse matrix is a sparse vector,
# and the right-hand side of a linear system is dense in every caller here. See `alloc_rhs`.
LinearProblem(A::AbstractMatrix) = LinearProblem(A, alloc_rhs(A))

function LinearProblem{T}(n::Integer, m::Integer) where {T}
    A = zeros(T, n, m)
    A .= T(NaN)
    LinearProblem(A)
end

LinearProblem{T}(n::Integer) where {T} = LinearProblem{T}(n, n)

LinearProblem(y::AbstractVector{T}) where {T} = LinearProblem{T}(length(y))

"""
    update!(ls::LinearProblem, A, b)

Set the [`rhs`](@ref) vector to `b` and the matrix stored in the [`LinearProblem`](@ref) `ls` to `A`.

!!! info
    Calling `update!` doesn't solve the [`LinearProblem`](@ref), you still have to call [`solve!`](@ref) in combination with a [`LinearSolver`](@ref).
"""
function update!(ls::LinearProblem{T}, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    copy_matrix!(matrix(ls), A)
    copy!(rhs(ls), b)
    ls
end

rhs(ls::LinearProblem) = ls.y
matrix(ls::LinearProblem) = ls.A

"""
    clear!(ls)

Write `NaN`s into `matrix(ls)` and `rhs(ls)`.

Here ls is a [`LinearProblem`](@ref).
"""
function clear!(ls::LinearProblem{T}) where {T}
    fill_nan!(matrix(ls))
    fill_nan!(rhs(ls))
    ls
end

"""
    initialize!(ls, x)

Initialize the [`LinearProblem`](@ref) `ls`.

This uses [`clear!(::LinearProblem)`](@ref).
"""
function initialize!(ls::LinearProblem, ::AbstractVector)
    clear!(ls)
    ls
end
