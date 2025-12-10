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
function update!(ls::LinearProblem{T}, A::AbstractMatrix{T}, y::AbstractVector{T}) where {T}
    update!(ls, A)
    update!(ls, y)
    ls
end

function update!(ls::LinearProblem{T}, A::AbstractMatrix{T}) where {T}
    ls.A .= A
    ls
end

function update!(ls::LinearProblem{T}, b::AbstractVector{T}) where {T}
    ls.y .= b
    ls
end

rhs(ls::LinearProblem) = ls.y
Base.Matrix(ls::LinearProblem)::AbstractMatrix = ls.A
Base.Vector(ls::LinearProblem)::AbstractVector = ls.y

"""
    clear!(ls)

Write `NaN`s into `Matrix(ls)` and `Vector(ls)`.
"""
function clear!(ls::LinearProblem{T}) where {T}
    Matrix(ls) .= T(NaN)
    Vector(ls) .= T(NaN)
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

"""
    NonlinearProblem

A `NonlinearProblem` describes ``F(x) = y``, where we want to solve for ``x`` and ``F`` is in nonlinear in general (also compare this to [`LinearProblem`](@ref) and [`OptimizerProblem`](@ref)).

!!! info
    `NonlinearProblem`s are used for *solvers* whereas `OptimizerProblem`s are their equivalent for *optimizers*.


# Keys
- `F`: accessed by calling `Function(nlp)`,
- `J::Union{Callable, Missing}`: accessed by calling `Jacobian(nlp)`,
"""
struct NonlinearProblem{T,TF<:Callable,TJ<:Union{Callable,Missing}} <: AbstractProblem
    F::TF
    J::TJ

    function NonlinearProblem(F::Callable, J::Union{Callable,Missing}, x::Tx, f::Tx=x) where {T,Tx<:AbstractArray{T}}
        new{T,typeof(F),typeof(J)}(F, J)
    end
end

NonlinearProblem{T}(F::Callable, J::Union{Callable, Missing}, n₁::Integer, n₂::Integer; kwargs...) where {T} = NonlinearProblem(F, J, zeros(T, n₁); kwargs...)
NonlinearProblem{T}(F::Callable, n₁::Integer, n₂::Integer; kwargs...) where {T} = NonlinearProblem{T}(F, missing, n₁, n₂)

@doc raw"""
    NonlinearProblem(F, x, f)

Set `jacobian` ``\gets`` `missing` and call the [`NonlinearProblem`](@ref) constructor.
"""
function NonlinearProblem(F::Callable, x::AbstractArray, f::AbstractArray)
    NonlinearProblem(F, missing, x, f)
end

"""
    value!(y, x, params)

Evaluate the [`NonlinearProblem`](@ref) at `x`.
"""
function value!(y::AbstractArray{T}, nlp::NonlinearProblem{T}, x::AbstractArray{T}, params) where {T}
    nlp.F(y, x, params)
end

# function value(nlp::NonlinearProblem{T}, x::AbstractVector{T}, params) where {T<:Number}
#     f = zero(value(nlp))
#     Function(nlp)(f, x, params)
#     f
# end

Base.Function(nlp::NonlinearProblem) = nlp.F


"""
    Jacobian(nlp::NonlinearProblem)

Return the *Jacobian function* stored in `nlp`.

!!! warn
   Note that this is different from the [`Jacobian`](@ref) used in the [`NonlinearSolver`](@ref)! There the [`Jacobian`](@ref) is a separate `struct`.
"""
Jacobian(nlp::NonlinearProblem) = nlp.J

function jacobian!(j::AbstractMatrix{T}, nlp::NonlinearProblem{T}, x::AbstractArray{T}, params) where {T}
    nlp.J(j, x, params)
end

function jacobian!(::AbstractMatrix{T}, ::NonlinearProblem{T, FT, Missing}, ::AbstractArray{T}, params) where {T, FT <: Callable}
    error("NonlinearSystem does not contain Jacobian.")
end