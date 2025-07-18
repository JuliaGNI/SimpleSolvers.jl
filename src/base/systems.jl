"""
    LinearSystem

A `LinearSystem` describes ``Ax = y``, where we want to solve for ``x``.

# Keys
- `A`
- `y`

# Constructors

A `LinearSystem` can be allocated by calling:

```julia
LinearSystem(A, y)
LinearSystem)(A)
LinearSystem(y)
LinearSystem{T}(n, m)
LinearSystem{T}(n)
```

Note that in any case the allocated system is initialized with `NaN`s:

```jldoctest; setup = :(using SimpleSolvers)
A = [1. 2. 3.; 4. 5. 6.; 7. 8. 9.]
y = [1., 2., 3.]
ls = LinearSystem(A, y)

# output

LinearSystem{Float64, Vector{Float64}, Matrix{Float64}}([NaN NaN NaN; NaN NaN NaN; NaN NaN NaN], [NaN, NaN, NaN])
```

In order to initialize the system with values, we have to call [`update!`](@ref):

```jldoctest; setup = :(using SimpleSolvers; A = [1. 2. 3.; 4. 5. 6.; 7. 8. 9.]; y = [1., 2., 3.]; ls = LinearSystem(A, y))
update!(ls, A, y)

# output

LinearSystem{Float64, Vector{Float64}, Matrix{Float64}}([1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0], [1.0, 2.0, 3.0])
```
"""
mutable struct LinearSystem{T,VT<:AbstractVector{T},AT<:AbstractMatrix{T}} <: AbstractProblem
    A::AT
    y::VT
    function LinearSystem(A::AT, y::VT) where {T<:Number,VT<:AbstractVector{T},AT<:AbstractMatrix{T}}
        @assert (length(y)) == size(A, 2)
        ls = new{T,VT,AT}(copy(A), copy(y))
        initialize!(ls, y)
        ls
    end
end

function LinearSystem(A::AbstractMatrix)
    y = alloc_x(A[:, 1])
    LinearSystem(A, y)
end

function LinearSystem{T}(n::Integer, m::Integer) where {T}
    A = zeros(T, n, m)
    A .= T(NaN)
    LinearSystem(A)
end

LinearSystem{T}(n::Integer) where {T} = LinearSystem{T}(n, n)

LinearSystem(y::AbstractVector{T}) where {T} = LinearSystem{T}(length(y))

"""
    update!(ls, A, y)

Set the [`rhs`](@ref) vector to `y` and the matrix stored in `ls` to `A`.

!!! info
    Calling `update!` doesn't solve the [`LinearSystem`](@ref), you still have to call `solve!` in combination with a [`LinearSolver`](@ref).
"""
function update!(ls::LinearSystem{T}, A::AbstractMatrix{T}, y::AbstractVector{T}) where {T}
    update!(ls, A)
    update!(ls, y)
    ls
end

function update!(ls::LinearSystem{T}, A::AbstractMatrix{T}) where {T}
    ls.A .= A
    ls
end

function update!(ls::LinearSystem{T}, b::AbstractVector{T}) where {T}
    ls.y .= b
    ls
end

rhs(ls::LinearSystem) = ls.y
Base.Matrix(ls::LinearSystem)::AbstractMatrix = ls.A
Base.Vector(ls::LinearSystem)::AbstractVector = ls.y

"""
    clear!(ls)

Write `NaN`s into `Matrix(ls)` and `Vector(ls)`.
"""
function clear!(ls::LinearSystem{T}) where {T}
    Matrix(ls) .= T(NaN)
    Vector(ls) .= T(NaN)
    ls
end

"""
    initialize!(ls, x)

Initialize the [`LinearSystem`](@ref) `ls`. See [`clear!(::LinearSystem)`](@ref).
"""
function initialize!(ls::LinearSystem, ::AbstractVector)
    clear!(ls)
    ls
end

"""
    NonlinearSystem

A `NonlinearSystem` describes ``F(x) = y``, where we want to solve for ``x`` and ``F`` is in nonlinear in general (also compare this to [`LinearSystem`](@ref) and [`MultivariateObjective`](@ref)).

!!! info
    `NonlinearSystem`s are used for *solvers* whereas `MultivariateObjective`s are their equivalent for *optimizers*.


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
mutable struct NonlinearSystem{T,FixedPoint,TF<:Callable,TJ<:Union{Jacobian{T},Missing},Tx<:AbstractVector{T},Tf<:AbstractVector{T},Tj<:AbstractMatrix{T}} <: AbstractProblem
    F::TF
    J::TJ

    f::Tf
    j::Tj

    x_f::Tx
    x_j::Tx

    f_calls::Int
    j_calls::Int

    function NonlinearSystem(F::Callable, J::Union{Jacobian,Missing},
        x::Tx,
        f::Tf;
        j::Tj=alloc_j(x, f),
        fixed_point::Bool=false) where {T,Tx<:AbstractArray{T},Tf,Tj<:AbstractArray{T}}
        applicable(F, f, x, NullParameters()) || error("The function needs to have the following signature: F(y, x, params).")
        nls = new{T,fixed_point,typeof(F),typeof(J),Tx,Tf,Tj}(F, J, alloc_x(f), j, alloc_x(x), alloc_x(x), 0, 0)
        initialize!(nls, x)
        nls
    end
end

"""
    NonlinearSystem(F, J!, x, f)
"""
function NonlinearSystem(F::Callable, J!::Callable,
    x::AbstractVector{T},
    f::AbstractVector{T};
    kwargs...) where {T<:Number}
    NonlinearSystem(F, JacobianFunction(J!, x), x, f; kwargs...)
end

"""
    NonlinearSystem(F, x, f)
"""
function NonlinearSystem(F::Callable, x::AbstractArray, f::AbstractArray; mode=:autodiff, kwargs...)
    mode == :autodiff || mode == :finite || error("If you want to use a manual Jacobian, please use a different constructor!")
    J = Jacobian(F, x; mode=mode, kwargs...)
    NonlinearSystem(F, J, x, f)
end

isFixedPointFormat(nls::NonlinearSystem{T,FP}) where {T,FP} = FP

function value!!(nls::NonlinearSystem{T}, x::AbstractArray{T}, params) where {T}
    f_argument(nls) .= x
    value(nls) .= value(nls, x, params)
end

function value(nls::NonlinearSystem{T}, x::AbstractVector{T}, params) where {T<:Number}
    nls.f_calls += 1
    f = zero(value(nls))
    Function(nls)(f, x, params)
    f
end

value(nls::NonlinearSystem) = nls.f

Base.Function(nls::NonlinearSystem) = nls.F

"""
    value!(nls::NonlinearSystem, x)

Check if `x` is not equal to `f_argument(nls)` and then apply [`value!!`](@ref). Else simply return `value(nls)`.
"""
function value!(nls::NonlinearSystem{T}, x::AbstractVector{T}, params) where {T<:Number}
    if x != f_argument(nls)
        value!!(nls, x, params)
    end
    value(nls)
end

"""
    jacobian(nls::NonlinearSystem)

Return the value of the jacobian stored in `nls` (instance of [`NonlinearSystem`](@ref)).
Like [`derivative`](@ref) for [`UnivariateObjective`](@ref) or [`gradient`](@ref) for [`MultivariateObjective`](@ref).

Also see [`Jacobian(::NonlinearSystem)`](@ref).
"""
jacobian(nls::NonlinearSystem) = nls.j

"""
    Jacobian(x, nls::NonlinearSystem)

Return the [`Jacobian`](@ref) stored in `nls`. Also see [`jacobian(::NonlinearSystem)`](@ref).
"""
Jacobian(nls::NonlinearSystem) = nls.J

function jacobian(nls::NonlinearSystem{T}, x::AbstractArray{T}, params) where {T<:Number}
    nls.j_calls += 1
    jacobian(x, Jacobian(nls), params)
end

"""
    jacobian!!(nls::NonlinearSystem, x)

Force the evaluation of the jacobian for a [`NonlinearSystem`](@ref).
Like [`derivative!!`](@ref) for [`UnivariateObjective`](@ref) or [`gradient!!`](@ref) for [`MultivariateObjective`](@ref).
"""
function jacobian!!(nls::NonlinearSystem{T}, x::AbstractArray{T}, params) where {T}
    copyto!(j_argument(nls), x)
    nls.j_calls += 1
    compute_jacobian!(jacobian(nls), x, Jacobian(nls), params)
    jacobian(nls)
end

"""
    jacobian!(nls::NonlinearSystem, x)

Compute the Jacobian of `nls` at `x` and store it in `jacobian(nls)`. Note that the evaluation of the Jacobian is not necessarily enforced here (unlike calling [`jacobian!!`](@ref)).
Like [`derivative!`](@ref) for [`MultivariateObjective`](@ref) and [`gradient!`](@ref) for [`UnivariateObjective`](@ref).
"""
function jacobian!(nls::NonlinearSystem{T}, x::AbstractArray{T}, params) where {T<:Number}
    if x != j_argument(nls)
        jacobian!!(nls, x, params)
    end
    jacobian(nls)
end

function _clear_f!(nls::NonlinearSystem{T}) where {T}
    nls.f_calls = 0
    f_argument(nls) .= T(NaN)
    value(nls) .= T(NaN)
    nothing
end

function _clear_j!(nls::NonlinearSystem{T}) where {T}
    nls.j_calls = 0
    j_argument(nls) .= T(NaN)
    jacobian(nls) .= T(NaN)

    nothing
end

"""
    clear!(nls::NonlinearSystem)

Similar to [`initialize!`](@ref), but with only one input argument.
"""
function clear!(nls::NonlinearSystem)
    _clear_f!(nls)
    _clear_j!(nls)
    nls
end

function initialize!(nls::NonlinearSystem, ::AbstractVector)
    clear!(nls)

    nls
end

"""
   f_argument(nls)

Return the argument that was last used for evaluating [`value!`](@ref) for the [`NonlinearSystem`](@ref) `nls`.
"""
f_argument(nls::NonlinearSystem) = nls.x_f

"""
   j_argument(nls)

Return the argument that was last used for evaluating [`jacobian!`](@ref) for the [`NonlinearSystem`](@ref) `nls`.
"""
j_argument(nls::NonlinearSystem) = nls.x_j

"""
    f_calls(nls)

Tell how many times `Function(nls)` has been called.

# Examples

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: NonlinearSystem, f_calls)
F(x) = tanh.(x)
x = [1., 2., 3.]
F!(y, x, params) = y .= F(x)
nls = NonlinearSystem(F!, x, F(x))

f_calls(nls)

# output

0
```

After calling [`value`](@ref) once we get:
```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: NonlinearSystem, f_calls; F(x) = tanh.(x); F!(y, x, params) = y .= F(x); x = [1., 2., 3.]; nls = NonlinearSystem(F!, x, F(x)))
value!(nls, x, nothing)

f_calls(nls)

# output

1
```
"""
f_calls(nls::NonlinearSystem) = nls.f_calls

"""
    j_calls(nls)

Like [`f_calls`](@ref) in relation to a [`NonlinearSystem`](@ref) `nls`, but for [`jacobian`](@ref) (or [`jacobian!`](@ref)).
"""
j_calls(nls::NonlinearSystem) = nls.j_calls

function update!(nls::NonlinearSystem{T}, x::AbstractVector{T}, params) where {T}
    value!(nls, x, params)
    jacobian!(nls, x, params)
    nls
end
