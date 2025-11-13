"""
    AbstractOptimizerProblem

An *optimizer problem* is a quantity to has to be made zero by a solver or minimized by an optimizer.

See [`LinesearchProblem`](@ref) and [`MultivariateOptimizerProblem`](@ref).
"""
abstract type AbstractOptimizerProblem{T <: Number} <: AbstractProblem end

Base.Function(obj::AbstractOptimizerProblem) = obj.F

clear!(::CT) where {CT <: Callable} = error("No method `clear!` implemented for type $(CT).")

"""
    value(obj::AbstractOptimizerProblem, x)

Evaluates the value at `x` (i.e. computes `obj.F(x)`).
"""
function value(obj::AbstractOptimizerProblem, x::Union{Number, AbstractArray{<:Number}})
    obj.f_calls += 1
    obj.F(x)
end

value(obj::AbstractOptimizerProblem) = obj.f

"""
    LinesearchProblem <: AbstractOptimizerProblem

Doesn't store `f`, `d`, `x_f` and `x_d` as well as `f_calls` and `d_calls`.

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
    obj.x_f = x
    obj.f = value(obj, x)
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
    MultivariateOptimizerProblem <: AbstractOptimizerProblem

Stores *gradients*. Also compare this to [`NonlinearProblem`](@ref).

The type of the *stored gradient* has to be a subtype of [`Gradient`](@ref).

# Functor

If `MultivariateOptimizerProblem` is called on a single function, the gradient is generated with [`GradientAutodiff`](@ref).
"""
mutable struct MultivariateOptimizerProblem{T, Tx <: AbstractVector{T}, TF <: Callable, TG <: Gradient{T}, Tf, Tg} <: AbstractOptimizerProblem{T}
    F::TF
    G::TG

    f::Tf
    g::Tg

    x_f::Tx
    x_g::Tx

    f_calls::Int
    g_calls::Int
end

function Base.show(io::IO, obj::MultivariateOptimizerProblem{T, Tx, TF, TG, Tf}) where {T, Tx, TF, TG, Tf <: Number}
    @printf io "MultivariateOptimizerProblem (for vector-valued quantities only the first component is printed):\n"
    @printf io "\n"
    @printf io "    f(x)              = %.2e %s" value(obj) "\n" 
    @printf io "    g(x)₁             = %.2e %s" gradient(obj)[1] "\n" 
    @printf io "    x_f₁              = %.2e %s" obj.x_f[1] "\n" 
    @printf io "    x_g₁              = %.2e %s" obj.x_g[1] "\n" 
    @printf io "    number of f calls = %s %s" obj.f_calls "\n" 
    @printf io "    number of g calls = %s %s" obj.g_calls "\n" 
end

function MultivariateOptimizerProblem(F::Callable, G::Gradient,
                               x::Tx;
                               f::Tf=eltype(x)(NaN),
                               g::Tg=alloc_g(x)) where {T, Tx<:AbstractArray{T}, Tf, Tg<:AbstractArray{T}}
    MultivariateOptimizerProblem{T, Tx, typeof(F), typeof(G), Tf, Tg}(F, G, f, g, alloc_x(x), alloc_x(x), 0, 0)
end

function MultivariateOptimizerProblem(F::Callable, G!::Callable,
                               x::AbstractArray{<:Number};
                               f::Number=eltype(x)(NaN),
                               g::AbstractArray{<:Number}=alloc_g(x))
    MultivariateOptimizerProblem(F, GradientFunction(G!, x), x; f = f, g = g)
end

function MultivariateOptimizerProblem(F::Callable, x::AbstractArray; kwargs...)
    G = Gradient(F, x; kwargs...)
    MultivariateOptimizerProblem(F, G, x; kwargs...)
end

MultivariateOptimizerProblem(F, G::Nothing, x::AbstractArray; kwargs...) = MultivariateOptimizerProblem(F, x; kwargs...)

function value!!(obj::MultivariateOptimizerProblem{T, Tx, TF, TG, Tf}, x::AbstractArray{<:Number}) where {T, Tx, TF, TG, Tf <: AbstractArray}
    f_argument(obj) .= x
    value(obj) .= value(obj, x)
end
function value!!(obj::MultivariateOptimizerProblem{T, Tx, TF, TG, Tf}, x::AbstractArray{<:Number}) where {T, Tx, TF, TG, Tf <: Number}
    f_argument(obj) .= x
    obj.f = value(obj, x)
end

"""
    gradient(x, obj::MultivariateOptimizerProblem)

Like [`derivative`](@ref), but for [`MultivariateOptimizerProblem`](@ref).
"""
gradient(obj::MultivariateOptimizerProblem) = obj.g

function gradient(obj::MultivariateOptimizerProblem, x::AbstractArray{<:Number})
    obj.g_calls += 1
    gradient(x, obj.G)
end

"""
    gradient!!(obj::MultivariateOptimizerProblem, x)

Like [`derivative!!`](@ref), but for [`MultivariateOptimizerProblem`](@ref).
"""
function gradient!!(obj::MultivariateOptimizerProblem, x::AbstractArray{<:Number})
    copyto!(obj.x_g, x)
    obj.g_calls += 1
    gradient!(gradient(obj), x, obj.G)
    gradient(obj)
end

"""
gradient!(obj::MultivariateOptimizerProblem, x)

Like [`derivative!`](@ref), but for [`MultivariateOptimizerProblem`](@ref).
"""
function gradient!(obj::MultivariateOptimizerProblem, x::AbstractArray{<:Number})
    if x != obj.x_g
        gradient!!(obj, x)
    end
    gradient(obj)
end

function _clear_f!(obj::MultivariateOptimizerProblem{T, Tx, TF, TG, Tf}) where {T, Tx, TF, TG, Tf <: Number}
    obj.f_calls = 0
    obj.f = T(NaN)
    f_argument(obj) .= T(NaN)
    nothing
end

function _clear_f!(obj::MultivariateOptimizerProblem{T, Tx, TF, TG, Tf}) where {T, Tx, TF, TG, Tf <: AbstractArray}
    obj.f_calls = 0
    obj.f .= T(NaN)
    f_argument(obj) .= T(NaN)
    nothing
end

function _clear_g!(obj::MultivariateOptimizerProblem{T}) where {T}
    obj.g_calls = 0
    obj.g .= T(NaN)
    g_argument(obj) .= T(NaN)
    nothing
end

"""
    clear!(obj)

Similar to [`initialize!`](@ref), but with only one input argument.
"""
function clear!(obj::MultivariateOptimizerProblem)
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
function update!(obj::MultivariateOptimizerProblem, x::AbstractVector)
    value!(obj, x)
    gradient!(obj, x)

    obj
end

f_argument(obj::AbstractOptimizerProblem) = obj.x_f
g_argument(obj::MultivariateOptimizerProblem) = obj.x_g

f_calls(o::AbstractOptimizerProblem) = error("f_calls is not implemented for $(summary(o)).")
f_calls(o::MultivariateOptimizerProblem) = o.f_calls

d_calls(o::AbstractOptimizerProblem) = error("d_calls is not implemented for $(summary(o)).")

g_calls(o::AbstractOptimizerProblem) = error("g_calls is not implemented for $(summary(o)).")
g_calls(o::MultivariateOptimizerProblem) = o.g_calls

Gradient(obj::MultivariateOptimizerProblem) = obj.G