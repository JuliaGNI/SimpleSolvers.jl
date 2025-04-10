"""
    AbstractObjective

An *objective* is a quantity to has to be made zero by a solver or minimized by an optimizer.

See [`AbstractUnivariateObjective`](@ref) and [`MultivariateObjective`](@ref).
"""
abstract type AbstractObjective end

"""
    AbstractUnivariateObjective <: AbstractObjective

A subtype of [`AbstractObjective`](@ref) that only depends on one variable. See [`UnivariateObjective`](@ref).
"""
abstract type AbstractUnivariateObjective <: AbstractObjective end

clear!(::CT) where {CT <: Callable} = error("No method `clear!` implemented for type $(CT).")

"""
    UnivariateObjective <: AbstractUnivariateObjective

# Keywords

It stores the following:
- `F`: objective
- `D`: derivative of objective
- `f`: cache for function output
- `d`: cache for derivative output
- `x_f`: x used to evaluate F (stored in f)
- `x_d`: x used to evaluate D (stored in d)
- `f_calls`: number of times `F` has been called
- `d_calls`: number of times `D` has been called

# Constructor

There are several constructors, the most generic (besides the default one) is:

```julia
UnivariateObjective(F, D, x; f, d)
```
Where no keys are inferred, except `x_f` and `x_d` (via [`alloc_f`](@ref) and [`alloc_d`](@ref)). `f_calls` and `d_calls` are set to zero.

The most general constructor (i.e. the one the needs the least specification) is:

```jldoctest; setup = :(using SimpleSolvers)
f(x::Number) = x ^ 2
UnivariateObjective(f, 1.)

# output

UnivariateObjective:\n
    f(x)              = NaN
    d(x)              = NaN
    x_f               = NaN
    x_d               = NaN
    number of f calls = 0
    number of d calls = 0
```
where `ForwardDiff` is used to generate the derivative of the (anonymous) function.

# Functor

The functor calls [`value!`](@ref).
"""
mutable struct UnivariateObjective{TF, TD, Tf, Td, Tx} <: AbstractUnivariateObjective
    F::TF
    D::TD

    f::Tf
    d::Td

    x_f::Tx
    x_d::Tx

    f_calls::Int
    d_calls::Int
end

function Base.show(io::IO, obj::UnivariateObjective)
    @printf io "UnivariateObjective:\n"
    @printf io "\n"
    @printf io "    f(x)              = %.2e %s" value(obj) "\n" 
    @printf io "    d(x)              = %.2e %s" derivative(obj) "\n" 
    @printf io "    x_f               = %.2e %s" obj.x_f "\n" 
    @printf io "    x_d               = %.2e %s" obj.x_d "\n" 
    @printf io "    number of f calls = %s %s" obj.f_calls "\n" 
    @printf io "    number of d calls = %s %s" obj.d_calls "\n" 
end

function UnivariateObjective(F::Callable, D::Callable, x::Number;
                             f::Real = alloc_f(x),
                             d::Number = alloc_d(x),
                             x_d::Number = alloc_x(x),
                             x_f::Number = alloc_x(x))
    UnivariateObjective(F, D, f, d, x_d, x_f, 0, 0)
end

function UnivariateObjective(F::Callable, x::Number; mode = :autodiff, kwargs...)
    @assert mode == :autodiff "Constructor for `UnivariateObjective` not defined for mode ≠ :autodiff."
    D = (x) -> ForwardDiff.derivative(F,x)
    UnivariateObjective(F, D, x; kwargs...)
end

UnivariateObjective(F::Callable, D::Nothing, x::Number; kwargs...) = UnivariateObjective(F, x; kwargs...)

"""
    value(obj::AbstractObjective, x)

Evaluates the objective value at `x` (i.e. computes `obj.F(x)`).

# Examples

```jldoctest
using SimpleSolvers

obj = UnivariateObjective(x::Number -> x^2, 1.)
value(obj, 2.)
obj.f_calls

# output

1
```

Note that the `f_calls` counter increased by one!

If `value` is called on `obj` (an [`AbstractObjective`](@ref)) without supplying `x` than the output of the last `obj.F` call is returned:

```jldoctest
using SimpleSolvers

obj = UnivariateObjective(x::Number -> x^2, 1.)
value(obj)

# output

NaN
```
In this example this is `NaN` since the function hasn't been called yet.
"""
function value(obj::AbstractObjective, x::Union{Number, AbstractArray{<:Number}})
    obj.f_calls += 1
    obj.F(x)
end

value(obj::AbstractObjective) = obj.f

"""
    value!!(obj::AbstractObjective, x)

Set `obj.x_f` to `x` and `obj.f` to `value(obj, x)` and return `value(obj)`.
"""
function value!!(obj::AbstractObjective, x::Number)
    obj.x_f = x
    obj.f = value(obj, x)
    value(obj)
end
function value!!(obj::AbstractObjective, x::AbstractArray{<:Number})
    obj.x_f .= x
    obj.f = value(obj, x)
    value(obj)
end

"""
    value!(obj::AbstractObjective, x)

Check if `x` is not equal to `obj.x_f` and then apply [`value!!`](@ref). Else simply return `value(obj)`.
"""
function value!(obj::AbstractObjective, x::Union{Number, AbstractArray{<:Number}})
    if x != obj.x_f
        value!!(obj, x)
    end
    value(obj)
end

"""
    derivative(obj::AbstractObjective, x)

Similar to [`value`](@ref), but for the derivative part (see [`UnivariateObjective`](@ref)).
"""
function derivative(obj::UnivariateObjective, x::Number)
    obj.d_calls += 1
    obj.D(x)
end

derivative(obj::UnivariateObjective) = obj.d

"""
    derivative!!(obj::AbstractObjective, x)

Similar to [`value!!`](@ref), but fo the derivative part (see [`UnivariateObjective`](@ref)).
"""
function derivative!!(obj::UnivariateObjective, x::Number)
    obj.x_d = x
    obj.d = derivative(obj, x)
    derivative(obj)
end
# function derivative!!(obj::AbstractObjective, x::AbstractArray{<:Number})
#     obj.x_d .= x
#     obj.d = derivative(obj, x)
#     derivative(obj)
# end

"""
    derivative!(obj, x)

Similar to [`value!`](@ref), but fo the derivative part (see [`UnivariateObjective`](@ref)).
"""
function derivative!(obj::UnivariateObjective, x::Number)
    if x != obj.x_d
        derivative!!(obj, x)
    end
    derivative(obj)
end

(obj::AbstractObjective)(x::Union{Number, AbstractArray{<:Number}}) = value!(obj, x)

function _clear_f!(obj::UnivariateObjective)
    obj.f_calls = 0
    obj.f = typeof(obj.f)(NaN)
    obj.x_f = typeof(obj.x_f)(NaN)
    nothing
end

function _clear_d!(obj::UnivariateObjective)
    obj.d_calls = 0
    obj.d = eltype(obj.d)(NaN)
    obj.x_d = eltype(obj.x_d)(NaN)
    nothing
end

"""
    clear!(obj)

Similar to [`initialize!`](@ref), but return `nothing`.
"""
function clear!(obj::AbstractUnivariateObjective)
    _clear_f!(obj)
    _clear_d!(obj)
    nothing
end

"""
    TemporaryUnivariateObjective <: AbstractUnivariateObjective

Like [`UnivariateObjective`](@ref) but doesn't store `f`, `d`, `x_f` and `x_d` as well as `f_calls` and `d_calls`.
"""
struct TemporaryUnivariateObjective{TF, TD} <: AbstractUnivariateObjective
    F::TF
    D::TD
end

TemporaryUnivariateObjective(f, d, x::Number) = TemporaryUnivariateObjective(f, d)

value(obj::TemporaryUnivariateObjective, x::Number) = obj.F(x)
value(obj::TemporaryUnivariateObjective) = error("TemporaryUnivariateObjective has to be called together with an x argument.")
derivative(obj::TemporaryUnivariateObjective, x::Number) = obj.D(x)

function value!(obj::TemporaryUnivariateObjective, x::Number)
    @warn "Calling value! on a TemporaryUnivariateObjective just calls value."
    value(obj, x)
end

function derivative!(obj::TemporaryUnivariateObjective, x::Number)
    @warn "Calling derivative! on a TemporaryUnivariateObjective just calls derivative."
    derivative(obj, x)
end

"""
    MultivariateObjective <: AbstractObjective

Like [`UnivariateObjective`](@ref), but stores *gradients* instead of *derivatives*.

The type of the *stored gradient* has to be a subtype of [`Gradient`](@ref).

# Functor

If `MultivariateObjective` is called on a single function, the gradient is generated with [`GradientAutodiff`](@ref).
"""
mutable struct MultivariateObjective{TF <: Callable, TG <: Gradient, Tf, Tg, Tx} <: AbstractObjective
    F::TF
    G::TG

    f::Tf
    g::Tg

    x_f::Tx
    x_g::Tx

    f_calls::Int
    g_calls::Int
end

function Base.show(io::IO, obj::MultivariateObjective)
    @printf io "MultivariateObjective (for vector-valued quantities only the first component is printed):\n"
    @printf io "\n"
    @printf io "    f(x)              = %.2e %s" value(obj) "\n" 
    @printf io "    g(x)₁             = %.2e %s" gradient(obj)[1] "\n" 
    @printf io "    x_f₁              = %.2e %s" obj.x_f[1] "\n" 
    @printf io "    x_g₁              = %.2e %s" obj.x_g[1] "\n" 
    @printf io "    number of f calls = %s %s" obj.f_calls "\n" 
    @printf io "    number of g calls = %s %s" obj.g_calls "\n" 
end

function MultivariateObjective(F::Callable, G::Gradient,
                               x::AbstractArray{<:Number};
                               f::Number=alloc_f(x),
                               g::AbstractArray{<:Number}=alloc_g(x))
    MultivariateObjective(F, G, f, g, alloc_x(x), alloc_x(x), 0, 0)
end

function MultivariateObjective(F::Callable, G!::Callable,
                               x::AbstractArray{<:Number};
                               f::Number=alloc_f(x),
                               g::AbstractArray{<:Number}=alloc_g(x))
    MultivariateObjective(F, GradientFunction(G!, x), x; f = f, g = g)
end

function MultivariateObjective(F::Callable, x::AbstractArray; kwargs...)
    G = GradientAutodiff(F, x)
    MultivariateObjective(F, G, x; kwargs...)
end

MultivariateObjective(F, G::Nothing, x::AbstractArray; kwargs...) = MultivariateObjective(F, x; kwargs...)

"""
    gradient(obj::MultivariateObjective, x)

Like [`derivative`](@ref), but for [`MultivariateObjective`](@ref), not [`UnivariateObjective`](@ref).
"""
gradient(obj::MultivariateObjective) = obj.g

function gradient(obj::MultivariateObjective, x::AbstractArray{<:Number})
    ḡ = similar(gradient(obj))
    obj.G(ḡ, x)
    obj.g_calls += 1
    ḡ
end

"""
    gradient(obj::MultivariateObjective, x)

Like [`derivative!!`](@ref), but for [`MultivariateObjective`](@ref), not [`UnivariateObjective`](@ref).
"""
function gradient!!(obj::MultivariateObjective, x::AbstractArray{<:Number})
    copyto!(obj.x_g, x)
    obj.G(obj.g, x)
    obj.g_calls += 1
    gradient(obj)
end

"""
gradient!(obj::MultivariateObjective, x)

Like [`derivative!`](@ref), but for [`MultivariateObjective`](@ref), not [`UnivariateObjective`](@ref).
"""
function gradient!(obj::MultivariateObjective, x)
    if x != obj.x_g
        gradient!!(obj, x)
    end
    gradient(obj)
end

function _clear_f!(obj::MultivariateObjective)
    obj.f_calls = 0
    obj.f = typeof(obj.f)(NaN)
    obj.x_f .= eltype(obj.x_f)(NaN)
    nothing
end

function _clear_g!(obj::MultivariateObjective)
    obj.g_calls = 0
    obj.g .= eltype(obj.g)(NaN)
    obj.x_g .= eltype(obj.x_g)(NaN)
    nothing
end

"""
    clear!(obj)

Similar to [`initialize!`](@ref), but return `nothing`.
"""
function clear!(obj::MultivariateObjective)
    _clear_f!(obj)
    _clear_g!(obj)
    nothing
end

f_calls(o::AbstractObjective) = error("f_calls is not implemented for $(summary(o)).")
f_calls(o::Union{UnivariateObjective, MultivariateObjective}) = o.f_calls

d_calls(o::AbstractObjective) = error("d_calls is not implemented for $(summary(o)).")
d_calls(o::UnivariateObjective) = o.d_calls

g_calls(o::AbstractObjective) = error("g_calls is not implemented for $(summary(o)).")
g_calls(o::MultivariateObjective) = o.g_calls