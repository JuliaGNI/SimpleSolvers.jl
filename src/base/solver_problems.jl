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
LinearProblem)(A)
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
- `f`: accessed by calling [`value`](@ref)`(nlp)`,
- `j`: accessed by calling [`jacobian`](@ref)`(nlp)`,
- `x_f`: accessed by calling [`f_argument`](@ref)`(nlp)`,
- `x_j`: accessed by calling [`j_argument`](@ref)`(nlp)`,
"""
struct NonlinearProblem{T,FixedPoint,TF<:Callable,TJ<:Union{Callable,Missing},Tx<:AbstractVector{T},Tf<:AbstractVector{T},Tj<:AbstractMatrix{T}} <: AbstractProblem
    F::TF
    J::TJ

    f::Tf
    j::Tj

    x_f::Tx
    x_j::Tx

    function NonlinearProblem(F::Callable, J::Union{Callable,Missing},
        x::Tx,
        f::Tf;
        j::Tj=alloc_j(x, f),
        fixed_point::Bool=false) where {T,Tx<:AbstractArray{T},Tf,Tj<:AbstractArray{T}}
        hasmethod(F, Tuple{Tf, Tx, OptionalParameters}) || error("The function needs to have the following signature: F(y, x, params).")
        nlp = new{T,fixed_point,typeof(F),typeof(J),Tx,Tf,Tj}(F, J, alloc_x(f), j, alloc_x(x), alloc_x(x))
        initialize!(nlp, x)
        nlp
    end
end

NonlinearProblem{T}(F::Callable, J::Union{Callable, Missing}, n₁::Integer, n₂::Integer; kwargs...) where {T} = NonlinearProblem(F, J, zeros(T, n₁), zeros(T, n₂); kwargs...)
NonlinearProblem{T}(F::Callable, n₁::Integer, n₂::Integer; kwargs...) where {T} = NonlinearProblem{T}(F, missing, n₁, n₂)

@doc raw"""
    NonlinearProblem(F, x, f)

Set `jacobian` ``\gets`` `missing` and call the [`NonlinearProblem`](@ref) constructor.
"""
function NonlinearProblem(F::Callable, x::AbstractArray, f::AbstractArray)
    NonlinearProblem(F, missing, x, f)
end

isFixedPointFormat(::NonlinearProblem{T,FP}) where {T,FP} = FP

function value!!(nlp::NonlinearProblem{T}, x::AbstractArray{T}, params) where {T}
    f_argument(nlp) .= x
    value(nlp) .= value(nlp, x, params)
end

function value(nlp::NonlinearProblem{T}, x::AbstractVector{T}, params) where {T<:Number}
    f = zero(value(nlp))
    Function(nlp)(f, x, params)
    f
end

value(nlp::NonlinearProblem) = nlp.f

Base.Function(nlp::NonlinearProblem) = nlp.F

"""
    value!(nlp::NonlinearProblem, x)

Check if `x` is not equal to `f_argument(nlp)` and then apply [`value!!`](@ref). Else simply return `value(nlp)`.
"""
function value!(nlp::NonlinearProblem{T}, x::AbstractVector{T}, params) where {T<:Number}
    if x != f_argument(nlp)
        value!!(nlp, x, params)
    end
    value(nlp)
end

"""
    jacobian(nlp::NonlinearProblem)

Return the value of the jacobian stored in `nlp` (instance of [`NonlinearProblem`](@ref)).
Like [`gradient`](@ref) for [`OptimizerProblem`](@ref).

Also see [`Jacobian(::NonlinearProblem)`](@ref).
"""
jacobian(nlp::NonlinearProblem) = nlp.j

"""
    Jacobian(nlp::NonlinearProblem)

Return the *Jacobian function* stored in `nlp`. Also see [`jacobian(::NonlinearProblem)`](@ref).

!!! warn
   Note that this is different from the [`Jacobian`](@ref) used in the [`NonlinearSolver`](@ref)! There the [`Jacobian`](@ref) is a separate `struct`.
"""
Jacobian(nlp::NonlinearProblem) = nlp.J

function jacobian(nlp::NonlinearProblem{T}, x::AbstractArray{T}, params) where {T<:Number}
    jacobian(x, Jacobian(nlp), params)
end

"""
    jacobian!!(nlp::NonlinearProblem, jacobian::Jacobian, x, prams)

Force the evaluation of the jacobian for a [`NonlinearProblem`](@ref).
Like [`gradient!!`](@ref) for [`OptimizerProblem`](@ref).
"""
function jacobian!!(nlp::NonlinearProblem{T}, jacobian_instance::Jacobian{T}, x::AbstractArray{T}, params) where {T}
    copyto!(j_argument(nlp), x)
    compute_jacobian!(jacobian(nlp), x, jacobian_instance, params)
    jacobian(nlp)
end

function jacobian!!(nlp::NonlinearProblem{T}, ::JacobianFunction{T}, x::AbstractArray{T}, params) where {T}
    copyto!(j_argument(nlp), x)
    Jacobian(nlp)(jacobian(nlp), x, params)
    jacobian(nlp)
end

function jacobian!!(::NonlinearProblem{T,FixedPoint,TF,Missing}, ::JacobianFunction{T}, x::AbstractArray{T}, params) where {T, FixedPoint, TF<:Callable}
    error("There is no analytic Jacobian stored in the system!")
end

"""
    jacobian!(nlp::NonlinearProblem, jacobian_instance, x, params)

Compute the Jacobian of `nlp` at `x` and store it in `jacobian(nlp)`. Note that the evaluation of the Jacobian is not necessarily enforced here (unlike calling [`jacobian!!`](@ref)).
Like `derivative!` for [`OptimizerProblem`](@ref).
"""
function jacobian!(nlp::NonlinearProblem{T}, jacobian_instance::Jacobian, x::AbstractArray{T}, params) where {T<:Number}
    if x != j_argument(nlp)
        jacobian!!(nlp, jacobian_instance, x, params)
    end
    jacobian(nlp)
end

function _clear_f!(nlp::NonlinearProblem{T}) where {T}
    f_argument(nlp) .= T(NaN)
    value(nlp) .= T(NaN)
    nothing
end

function _clear_j!(nlp::NonlinearProblem{T}) where {T}
    j_argument(nlp) .= T(NaN)
    jacobian(nlp) .= T(NaN)

    nothing
end

"""
    clear!(nlp::NonlinearProblem)

Similar to [`initialize!`](@ref), but with only one input argument.
"""
function clear!(nlp::NonlinearProblem)
    _clear_f!(nlp)
    _clear_j!(nlp)
    nlp
end

function initialize!(nlp::NonlinearProblem, ::AbstractVector)
    clear!(nlp)

    nlp
end

"""
   f_argument(nlp)

Return the argument that was last used for evaluating [`value!`](@ref) for the [`NonlinearProblem`](@ref) `nlp`.
"""
f_argument(nlp::NonlinearProblem) = nlp.x_f

"""
   j_argument(nlp)

Return the argument that was last used for evaluating [`jacobian!`](@ref) for the [`NonlinearProblem`](@ref) `nlp`.
"""
j_argument(nlp::NonlinearProblem) = nlp.x_j

function update!(nlp::NonlinearProblem{T}, jacobian::Jacobian{T}, x::AbstractVector{T}, params) where {T}
    value!(nlp, x, params)
    jacobian!(nlp, jacobian, x, params)
    nlp
end
