"""
    AbstractOptimizerProblem

An *optimizer problem* is a quantity to has to be made zero by a solver or minimized by an optimizer.

See [`LinesearchProblem`](@ref) and [`OptimizerProblem`](@ref).
"""
abstract type AbstractOptimizerProblem{T <: Number} <: AbstractProblem end

Base.Function(obj::AbstractOptimizerProblem) = obj.F

clear!(::CT) where {CT <: Callable} = error("No method `clear!` implemented for type $(CT).")

"""
    value(obj::AbstractOptimizerProblem, x)

Evaluates the value at `x` (i.e. computes `obj.F(x)`).
"""
function value(obj::AbstractOptimizerProblem, x::Union{Number, AbstractArray{<:Number}})
    obj.F(x)
end

value(obj::AbstractOptimizerProblem) = obj.f

"""
    LinesearchProblem <: AbstractOptimizerProblem

Doesn't store `f`, `d`, `x_f` and `x_d`.

In practice `LinesearchProblem`s are allocated by calling [`linesearch_problem`](@ref).

# Constructors

!!! warn "Calling line search problems"
    Below we show a few constructors that can be used to allocate `LinesearchProblem`s. Note however that in practice one probably should not do that and instead call `linesearch_problem`.

```jldoctest; setup = :(using SimpleSolvers: LinesearchProblem, compute_new_iterate)
f(x) = x^2 - 1
g(x) = 2x
δx(x) = - g(x) / 2
x₀ = 3.
_f(α) = f(compute_new_iterate(x₀, α, δx(x₀)))
_d(α) = g(compute_new_iterate(x₀, α, δx(x₀)))
ls_obj = LinesearchProblem{typeof(x₀)}(_f, _d)

# output

LinesearchProblem{Float64, typeof(_f), typeof(_d)}(_f, _d)
```

Alternatively one can also do:

```jldoctest; setup = :(using SimpleSolvers: LinesearchProblem, compute_new_iterate; f(x) = x^2 - 1; g(x) = 2x; δx(x) = - g(x) / 2; x₀ = 3.; _f(α) = f(compute_new_iterate(x₀, α, δx(x₀))); _d(α) = g(compute_new_iterate(x₀, α, δx(x₀))))
ls_obj = LinesearchProblem(_f, _d, x₀)

# output

LinesearchProblem{Float64, typeof(_f), typeof(_d)}(_f, _d)
```

Here we wrote `ls_obj` to mean *line search problem*.
"""
struct LinesearchProblem{Tx <: Number, TF, TD} <: AbstractOptimizerProblem{Tx}
    F::TF
    D::TD
end

"""
    value!!(obj::AbstractOptimizerProblem, x)

Set `obj.x_f` to `x` and `obj.f` to `value(obj, x)` and return `value(obj)`.
"""
function value!!(obj::LinesearchProblem, x::Number)
    copyto!(obj.x_f, x)
    copyto!(obj.f, value(obj, x))
end

"""
    value!(obj::AbstractOptimizerProblem, x)

Check if `x` is not equal to `obj.x_f` and then apply [`value!!`](@ref). Else simply return `value(obj)`.
"""
function value!(obj::AbstractOptimizerProblem, x::Union{Number, AbstractArray{<:Number}})
    if x != obj.x_f
        value!!(obj, x)
    end
    value(obj)
end

(obj::AbstractOptimizerProblem)(x::Union{Number, AbstractArray{<:Number}}) = value!(obj, x)

LinesearchProblem{Tx}(f, d) where {Tx <: Number} = LinesearchProblem{Tx, typeof(f), typeof(d)}(f, d)

LinesearchProblem(f, d, ::Tx=zero(Float64)) where {Tx <: Number} = LinesearchProblem{Tx}(f, d)

value(obj::LinesearchProblem, x::Number) = obj.F(x)
value(::LinesearchProblem) = error("LinesearchProblem has to be called together with an x argument.")
derivative(obj::LinesearchProblem, x::Number) = obj.D(x)

function value!(obj::LinesearchProblem, x::Number)
    value(obj, x)
end

function derivative!(obj::LinesearchProblem, x::Number)
    derivative(obj, x)
end

"""
    OptimizerProblem <: AbstractOptimizerProblem

Stores *gradients*. Also compare this to [`NonlinearProblem`](@ref).

The type of the *stored gradient* has to be a subtype of [`Gradient`](@ref).

# Functor

If `OptimizerProblem` is called on a single function, the gradient is generated with [`GradientAutodiff`](@ref).
"""
mutable struct OptimizerProblem{T, Tx <: AbstractVector{T}, TF <: Callable, TG <: Union{Callable, Missing}, Tf, Tg} <: AbstractOptimizerProblem{T}
    F::TF
    G::TG

    f::Tf
    g::Tg

    x_f::Tx
    x_g::Tx
end

function Base.show(io::IO, obj::OptimizerProblem{T, Tx, TF, TG, Tf}) where {T, Tx, TF, TG, Tf <: Number}
    @printf io "OptimizerProblem (for vector-valued quantities only the first component is printed):\n"
    @printf io "\n"
    @printf io "    f(x)              = %.2e %s" value(obj) "\n" 
    @printf io "    g(x)₁             = %.2e %s" gradient(obj)[1] "\n" 
    @printf io "    x_f₁              = %.2e %s" obj.x_f[1] "\n" 
    @printf io "    x_g₁              = %.2e %s" obj.x_g[1] "\n" 
end

function OptimizerProblem(F::Callable, G::TG,
                               x::Tx;
                               f::T=T(NaN),
                               g::Tg=alloc_g(x)) where {T<:Number, Tx<:AbstractArray{T}, Tg<:AbstractArray{T}, TG <: Union{Callable, Missing}}
    OptimizerProblem{T, Tx, typeof(F), TG, T, Tg}(F, G, f, g, alloc_x(x), alloc_x(x))
end

function OptimizerProblem(F::Callable, x::Tx;
                               f::Number=T(NaN),
                               g::Tg=alloc_g(x), gradient = missing) where {T<:Number, Tx<:AbstractArray{T}, Tg<:AbstractArray{T}}
    OptimizerProblem(F, gradient, x; f = f, g = g)
end

function value!!(obj::OptimizerProblem{T, Tx, TF, TG, Tf}, x::AbstractArray{<:Number}) where {T, Tx, TF, TG, Tf <: AbstractArray}
    copyto!(f_argument(obj), x)
    copyto!(value(obj), value(obj, x))
end
function value!!(obj::OptimizerProblem{T, Tx, TF, TG, Tf}, x::AbstractArray{<:Number}) where {T, Tx, TF, TG, Tf <: Number}
    copyto!(f_argument(obj), x)
    obj.f = value(obj, x)
end

"""
    gradient(x, obj::OptimizerProblem)

Like `derivative`, but for [`OptimizerProblem`](@ref).
"""
gradient(obj::OptimizerProblem) = obj.g

"""
    gradient!!(obj::OptimizerProblem, gradient_instance, x)

Like `derivative!!`, but for [`OptimizerProblem`](@ref).
"""
function gradient!!(obj::OptimizerProblem, gradient_instance::Gradient, x::AbstractArray)
    copyto!(g_argument(obj), x)
    gradient!(gradient(obj), gradient_instance, x)
    gradient(obj)
end

function gradient!!(obj::OptimizerProblem{T, Tx, TF, TG}, ::GradientFunction, x::Tx) where {T, Tx<:AbstractVector{T}, TF<:Callable, TG<:Callable}
    copyto!(g_argument(obj), x)
    Gradient(obj)(gradient(obj), x)
    gradient(obj)
end

function gradient!!(::OptimizerProblem{T, Tx, TF, Missing}, ::GradientFunction, ::AbstractArray{<:Number}) where {T, Tx<:AbstractVector{T}, TF<:Callable}
    error("There is no analytic gradient stored in the problem!")    
end

"""
gradient!(obj::OptimizerProblem, x)

Like `derivative!`, but for [`OptimizerProblem`](@ref).
"""
function gradient!(obj::OptimizerProblem, gradient_instance::Gradient, x::AbstractArray{<:Number})
    if x != obj.x_g
        gradient!!(obj, gradient_instance, x)
    end
    gradient(obj)
end

function _clear_f!(obj::OptimizerProblem{T, Tx, TF, TG, Tf}) where {T, Tx, TF, TG, Tf <: Number}
    obj.f = T(NaN)
    f_argument(obj) .= T(NaN)
    nothing
end

function _clear_f!(obj::OptimizerProblem{T, Tx, TF, TG, Tf}) where {T, Tx, TF, TG, Tf <: AbstractArray}
    obj.f .= T(NaN)
    f_argument(obj) .= T(NaN)
    nothing
end

function _clear_g!(obj::OptimizerProblem{T}) where {T}
    obj.g .= T(NaN)
    g_argument(obj) .= T(NaN)
    nothing
end

"""
    clear!(obj)

Similar to [`initialize!`](@ref), but with only one input argument.
"""
function clear!(obj::OptimizerProblem)
    _clear_f!(obj)
    _clear_g!(obj)
    obj
end

function initialize!(obj::AbstractOptimizerProblem, ::AbstractVector)
    clear!(obj)
end

"""
    update!(obj, x)

Call [`value!`](@ref) and [`gradient!`](@ref) on `obj`.
"""
function update!(obj::OptimizerProblem, gradient_instance::Gradient, x::AbstractVector)
    value!(obj, x)
    gradient!(obj, gradient_instance, x)

    obj
end

f_argument(obj::AbstractOptimizerProblem) = obj.x_f
g_argument(obj::OptimizerProblem) = obj.x_g

Gradient(obj::OptimizerProblem) = obj.G