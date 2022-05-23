
abstract type AbstractObjective end

clear!(::Function) = nothing

mutable struct UnivariateObjective{TF, TD, Tf, Td, Tx} <: AbstractObjective
    F::TF     # objective
    D::TD     # derivative of objective

    f::Tf     # cache for f output
    d::Td     # cache for d output

    x_f::Tx   # x used to evaluate F (stored in f)
    x_d::Tx   # x used to evaluate D (stored in d)

    f_calls::Int
    d_calls::Int
end

function UnivariateObjective(F::Callable, D::Callable, x::Number;
                             f::Real = alloc_f(x),
                             d::Number = alloc_d(x))
    UnivariateObjective(F, D, f, d, alloc_x(x), alloc_x(x), 0, 0)
end

function UnivariateObjective(F::Callable, x::Number; kwargs...)
    D = (x) -> ForwardDiff.derivative(F,x)
    UnivariateObjective(F, D, x; kwargs...)
end

UnivariateObjective(F::Callable, D::Nothing, x::Number; kwargs...) = UnivariateObjective(F, x; kwargs...)


"""
Evaluates the objective value at `x`.
Returns `f(x)`, but does *not* store the value in `obj.F`
"""
function value(obj::UnivariateObjective, x)
    obj.f_calls += 1
    return obj.F(x)
end

"Get the most recently evaluated objective value of `obj`."
value(obj::UnivariateObjective) = obj.f

"""
Force (re-)evaluation of the objective value at `x`.
Returns `f(x)` and stores the value in `obj.F`
"""
function value!!(obj::UnivariateObjective, x)
    obj.x_f = x
    obj.f = obj.F(x)
    obj.f_calls += 1
    value(obj)
end

"""
Evaluates the objective value at `x`.
Returns `f(x)` and stores the value in `obj.F`
"""
function value!(obj::UnivariateObjective, x)
    if x != obj.x_f
        value!!(obj, x)
    end
    value(obj)
end


"""
Evaluates the derivative of the objective at `x`.
Returns `f'(x)`, but does *not* store the derivative in `obj.D`
"""
function derivative(obj::UnivariateObjective, x)
    obj.d_calls += 1
    return obj.D(x)
end

"Get the most recently evaluated derivative of the objective of `obj`."
derivative(obj::UnivariateObjective) = obj.d

"""
Force (re-)evaluation of the derivative of the objective at `x`.
Returns `f'(x)` and stores the derivative in `obj.D`
"""
function derivative!!(obj::UnivariateObjective, x)
    obj.x_d = x
    obj.d = obj.D(x)
    obj.d_calls += 1
    derivative(obj)
end

"""
Evaluates the derivative of the objective at `x`.
Returns `f'(x)` and stores the derivative in `obj.D`
"""
function derivative!(obj::UnivariateObjective, x)
    if x != obj.x_d
        derivative!!(obj, x)
    end
    derivative(obj)
end


(obj::UnivariateObjective)(x) = value!(obj, x)


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

function clear!(obj::UnivariateObjective)
    _clear_f!(obj)
    _clear_d!(obj)
    nothing
end



mutable struct MultivariateObjective{TF, TG, Tf, Tg, Tx} <: AbstractObjective
    F::TF
    G::TG

    f::Tf
    g::Tg

    x_f::Tx
    x_g::Tx

    f_calls::Int
    g_calls::Int
end

function MultivariateObjective(F, G,
                               x::AbstractArray;
                               f::Real = alloc_f(x),
                               g::AbstractArray = alloc_g(x))
    MultivariateObjective(F, G, f, g, alloc_x(x), alloc_x(x), 0, 0)
end

function MultivariateObjective(F, x::AbstractArray; kwargs...)
    G = GradientAutodiff(F, x)
    MultivariateObjective(F, G, x; kwargs...)
end

MultivariateObjective(F, G::Nothing, x::AbstractArray; kwargs...) = MultivariateObjective(F, x; kwargs...)

"""
Evaluates the objective value at `x`.
Returns `f(x)`, but does *not* store the value in `obj.f`
"""
function value(obj::MultivariateObjective, x)
    obj.f_calls += 1
    return obj.F(x)
end

"Get the most recently evaluated objective value of `obj`."
value(obj::MultivariateObjective) = obj.f

"""
Force (re-)evaluation of the objective at `x`.
Returns `f(x)` and stores the value in `obj.f`
"""
function value!!(obj::MultivariateObjective, x)
    copyto!(obj.x_f, x)
    obj.f = obj.F(x)
    obj.f_calls += 1
    value(obj)
end

"""
Evaluates the objective at `x`.
Returns `f(x)` and stores the value in `obj.f`
"""
function value!(obj::MultivariateObjective, x)
    if x != obj.x_f
        value!!(obj, x)
    end
    value(obj)
end


"Get the most recently evaluated gradient of `obj`."
gradient(obj::MultivariateObjective) = obj.g

"""
Evaluates the gradient at `x`.
This does *not* update `obj.g` or `obj.x_g`.
"""
function gradient(obj::MultivariateObjective, x)
    ḡ = copy(obj.g)
    obj.G(ḡ, x)
    obj.g_calls += 1
    return ḡ
end

"""
Force (re-)evaluation of the gradient at `x`.
Returns ∇f(x) and stores the value in `obj.g`.
"""
function gradient!!(obj::MultivariateObjective, x)
    copyto!(obj.x_g, x)
    obj.G(obj.g, x)
    obj.g_calls += 1
    gradient(obj)
end

"""
Evaluates the gradient at `x`.
Returns ∇f(x) and stores the value in `obj.g`.
"""
function gradient!(obj::MultivariateObjective, x)
    if x != obj.x_g
        gradient!!(obj, x)
    end
    gradient(obj)
end


(obj::MultivariateObjective)(x) = value!(obj, x)


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

function clear!(obj::MultivariateObjective)
    _clear_f!(obj)
    _clear_g!(obj)
    nothing
end



f_calls(o::AbstractObjective) = error("f_calls is not implemented for $(summary(o)).")
f_calls(o::UnivariateObjective) = o.f_calls
f_calls(o::MultivariateObjective) = o.f_calls

d_calls(o::AbstractObjective) = error("d_calls is not implemented for $(summary(o)).")
d_calls(o::UnivariateObjective) = o.d_calls

g_calls(o::AbstractObjective) = error("g_calls is not implemented for $(summary(o)).")
g_calls(o::MultivariateObjective) = o.g_calls
