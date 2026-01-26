"""
Encompasses the [`NoLinearProblem`](@ref) and the [`LinearProblem`](@ref).
"""
abstract type AbstractLinearProblem <: AbstractProblem end

"""
A *dummy linear system* used for the *fixed point iterator* ([`PicardMethod`](@ref)).
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

Note that in any case the allocated system is initialized with `NaN`s:

```jldoctest; setup = :(using SimpleSolvers)
A = [1. 2. 3.; 4. 5. 6.; 7. 8. 9.]
y = [1., 2., 3.]
ls = LinearProblem(A, y)

# output

LinearProblem{Float64, Vector{Float64}, Matrix{Float64}}([NaN NaN NaN; NaN NaN NaN; NaN NaN NaN], [NaN, NaN, NaN])
```

In order to initialize the system with values, we have to call [`update!`](@ref):

```jldoctest; setup = :(using SimpleSolvers; A = [1. 2. 3.; 4. 5. 6.; 7. 8. 9.]; y = [1., 2., 3.]; ls = LinearProblem(A, y))
update!(ls, A, y)

# output

LinearProblem{Float64, Vector{Float64}, Matrix{Float64}}([1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0], [1.0, 2.0, 3.0])
```
"""
mutable struct LinearProblem{T,VT<:AbstractVector{T},AT<:AbstractMatrix{T}} <: AbstractLinearProblem
    A::AT
    y::VT
    function LinearProblem(A::AT, y::VT) where {T<:Number,VT<:AbstractVector{T},AT<:AbstractMatrix{T}}
        @assert (length(y)) == size(A, 2)
        ls = new{T,VT,AT}(copy(A), copy(y))
        initialize!(ls, y)
        ls
    end
end

function LinearProblem(A::AbstractMatrix)
    y = alloc_x(A[:, 1])
    LinearProblem(A, y)
end

function LinearProblem{T}(n::Integer, m::Integer) where {T}
    A = zeros(T, n, m)
    A .= T(NaN)
    LinearProblem(A)
end

LinearProblem{T}(n::Integer) where {T} = LinearProblem{T}(n, n)

LinearProblem(y::AbstractVector{T}) where {T} = LinearProblem{T}(length(y))

"""
    update!(ls, A, y)

Set the [`rhs`](@ref) vector to `y` and the matrix stored in `ls` to `A`.

!!! info
    Calling `update!` doesn't solve the [`LinearProblem`](@ref), you still have to call `solve!` in combination with a [`LinearSolver`](@ref).
"""
function update!(ls::LinearProblem{T}, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    copy!(matrix(ls), A)
    copy!(rhs(ls), b)
    ls
end

rhs(ls::LinearProblem) = ls.y
matrix(ls::LinearProblem)::AbstractMatrix = ls.A

"""
    clear!(ls)

Write `NaN`s into `Matrix(ls)` and `Vector(ls)`.
"""
function clear!(ls::LinearProblem{T}) where {T}
    matrix(ls) .= T(NaN)
    rhs(ls) .= T(NaN)
    ls
end

"""
    initialize!(ls, x)

Initialize the [`LinearProblem`](@ref) `ls`. See [`clear!(::LinearProblem)`](@ref).
"""
function initialize!(ls::LinearProblem, ::AbstractVector)
    clear!(ls)
    ls
end
