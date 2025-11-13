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

A `NonlinearProblem` describes ``F(x) = y``, where we want to solve for ``x`` and ``F`` is in nonlinear in general (also compare this to [`LinearProblem`](@ref) and [`MultivariateOptimizerProblem`](@ref)).

!!! info
    `NonlinearProblem`s are used for *solvers* whereas `MultivariateOptimizerProblem`s are their equivalent for *optimizers*.


# Keys
- `F`: accessed by calling `Function(nls)`,
- `J::`[`Jacobian`](@ref): accessed by calling `Jacobian(nls)`,
- `f`: accessed by calling [`value`](@ref)`(nls)`,
- `j`: accessed by calling [`jacobian`](@ref)`(nls)`,
- `x_f`: accessed by calling [`f_argument`](@ref)`(nls)`,
- `x_j`: accessed by calling [`j_argument`](@ref)`(nls)`,
- `f_calls`: accessed by calling [`f_calls`](@ref)`(nls)`,
- `j_calls`: accessed by calling [`j_calls`](@ref)`(nls)`.

"""
mutable struct NonlinearProblem{T,FixedPoint,TF<:Callable,TJ<:Union{Jacobian{T},Missing},Tx<:AbstractVector{T},Tf<:AbstractVector{T},Tj<:AbstractMatrix{T}} <: AbstractProblem
    F::TF
    J::TJ

    f::Tf
    j::Tj

    x_f::Tx
    x_j::Tx

    f_calls::Int
    j_calls::Int

    function NonlinearProblem(F::Callable, J::Union{Jacobian,Missing},
        x::Tx,
        f::Tf;
        j::Tj=alloc_j(x, f),
        fixed_point::Bool=false) where {T,Tx<:AbstractArray{T},Tf,Tj<:AbstractArray{T}}
        hasmethod(F, Tuple{Tf, Tx, OptionalParameters}) || error("The function needs to have the following signature: F(y, x, params).")
        nls = new{T,fixed_point,typeof(F),typeof(J),Tx,Tf,Tj}(F, J, alloc_x(f), j, alloc_x(x), alloc_x(x), 0, 0)
        initialize!(nls, x)
        nls
    end
end

"""
    NonlinearProblem(F, J!, x, f)
"""
function NonlinearProblem(F::Callable, J!::Callable,
    x::AbstractVector{T},
    f::AbstractVector{T};
    kwargs...) where {T<:Number}
    NonlinearProblem(F, JacobianFunction(J!, x), x, f; kwargs...)
end

"""
    NonlinearProblem(F, x, f)
"""
function NonlinearProblem(F::Callable, x::AbstractArray, f::AbstractArray; mode=:autodiff, kwargs...)
    mode == :autodiff || mode == :finite || error("If you want to use a manual Jacobian, please use a different constructor!")
    J = Jacobian(F, x; mode=mode, kwargs...)
    NonlinearProblem(F, J, x, f)
end

isFixedPointFormat(nls::NonlinearProblem{T,FP}) where {T,FP} = FP

function value!!(nls::NonlinearProblem{T}, x::AbstractArray{T}, params) where {T}
    f_argument(nls) .= x
    value(nls) .= value(nls, x, params)
end

function value(nls::NonlinearProblem{T}, x::AbstractVector{T}, params) where {T<:Number}
    nls.f_calls += 1
    f = zero(value(nls))
    Function(nls)(f, x, params)
    f
end

value(nls::NonlinearProblem) = nls.f

Base.Function(nls::NonlinearProblem) = nls.F

"""
    value!(nls::NonlinearProblem, x)

Check if `x` is not equal to `f_argument(nls)` and then apply [`value!!`](@ref). Else simply return `value(nls)`.
"""
function value!(nls::NonlinearProblem{T}, x::AbstractVector{T}, params) where {T<:Number}
    if x != f_argument(nls)
        value!!(nls, x, params)
    end
    value(nls)
end

"""
    jacobian(nls::NonlinearProblem)

Return the value of the jacobian stored in `nls` (instance of [`NonlinearProblem`](@ref)).
Like [`gradient`](@ref) for [`MultivariateOptimizerProblem`](@ref).

Also see [`Jacobian(::NonlinearProblem)`](@ref).
"""
jacobian(nls::NonlinearProblem) = nls.j

"""
    Jacobian(x, nls::NonlinearProblem)

Return the [`Jacobian`](@ref) stored in `nls`. Also see [`jacobian(::NonlinearProblem)`](@ref).
"""
Jacobian(nls::NonlinearProblem) = nls.J

function jacobian(nls::NonlinearProblem{T}, x::AbstractArray{T}, params) where {T<:Number}
    nls.j_calls += 1
    jacobian(x, Jacobian(nls), params)
end

"""
    jacobian!!(nls::NonlinearProblem, x)

Force the evaluation of the jacobian for a [`NonlinearProblem`](@ref).
Like [`gradient!!`](@ref) for [`MultivariateOptimizerProblem`](@ref).
"""
function jacobian!!(nls::NonlinearProblem{T}, x::AbstractArray{T}, params) where {T}
    copyto!(j_argument(nls), x)
    nls.j_calls += 1
    compute_jacobian!(jacobian(nls), x, Jacobian(nls), params)
    jacobian(nls)
end

"""
    jacobian!(nls::NonlinearProblem, x)

Compute the Jacobian of `nls` at `x` and store it in `jacobian(nls)`. Note that the evaluation of the Jacobian is not necessarily enforced here (unlike calling [`jacobian!!`](@ref)).
Like [`derivative!`](@ref) for [`MultivariateOptimizerProblem`](@ref).
"""
function jacobian!(nls::NonlinearProblem{T}, x::AbstractArray{T}, params) where {T<:Number}
    if x != j_argument(nls)
        jacobian!!(nls, x, params)
    end
    jacobian(nls)
end

function _clear_f!(nls::NonlinearProblem{T}) where {T}
    nls.f_calls = 0
    f_argument(nls) .= T(NaN)
    value(nls) .= T(NaN)
    nothing
end

function _clear_j!(nls::NonlinearProblem{T}) where {T}
    nls.j_calls = 0
    j_argument(nls) .= T(NaN)
    jacobian(nls) .= T(NaN)

    nothing
end

"""
    clear!(nls::NonlinearProblem)

Similar to [`initialize!`](@ref), but with only one input argument.
"""
function clear!(nls::NonlinearProblem)
    _clear_f!(nls)
    _clear_j!(nls)
    nls
end

function initialize!(nls::NonlinearProblem, ::AbstractVector)
    clear!(nls)

    nls
end

"""
   f_argument(nls)

Return the argument that was last used for evaluating [`value!`](@ref) for the [`NonlinearProblem`](@ref) `nls`.
"""
f_argument(nls::NonlinearProblem) = nls.x_f

"""
   j_argument(nls)

Return the argument that was last used for evaluating [`jacobian!`](@ref) for the [`NonlinearProblem`](@ref) `nls`.
"""
j_argument(nls::NonlinearProblem) = nls.x_j

"""
    f_calls(nls)

Tell how many times `Function(nls)` has been called.

# Examples

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: NonlinearProblem, f_calls)
F(x) = tanh.(x)
x = [1., 2., 3.]
F!(y, x, params) = y .= F(x)
nls = NonlinearProblem(F!, x, F(x))

f_calls(nls)

# output

0
```

After calling [`value`](@ref) once we get:
```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: NonlinearProblem, f_calls; F(x) = tanh.(x); F!(y, x, params) = y .= F(x); x = [1., 2., 3.]; nls = NonlinearProblem(F!, x, F(x)))
value!(nls, x, nothing)

f_calls(nls)

# output

1
```
"""
f_calls(nls::NonlinearProblem) = nls.f_calls

"""
    j_calls(nls)

Like [`f_calls`](@ref) in relation to a [`NonlinearProblem`](@ref) `nls`, but for [`jacobian`](@ref) (or [`jacobian!`](@ref)).
"""
j_calls(nls::NonlinearProblem) = nls.j_calls

function update!(nls::NonlinearProblem{T}, x::AbstractVector{T}, params) where {T}
    value!(nls, x, params)
    jacobian!(nls, x, params)
    nls
end
