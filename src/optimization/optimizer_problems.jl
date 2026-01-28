"""
    AbstractOptimizerProblem

An *optimizer problem* is a quantity to has to be made zero by a solver or minimized by an optimizer.

See [`LinesearchProblem`](@ref) and [`OptimizerProblem`](@ref).
"""
abstract type AbstractOptimizerProblem{T <: Number} <: AbstractProblem end

Base.Function(obj::AbstractOptimizerProblem) = obj.F

"""
    value(obj::AbstractOptimizerProblem, x)

Evaluates the value at `x` (i.e. computes `obj.F(x)`).
"""
function value(obj::AbstractOptimizerProblem, x::Union{Number, AbstractArray{<:Number}})
    obj.F(x)
end

"""
    LinesearchProblem <: AbstractOptimizerProblem

Doesn't store `f`, `d`, `x_f` and `x_d`.

In practice `LinesearchProblem`s are allocated by calling [`linesearch_problem`](@ref).

# Constructors

!!! warning "Calling line search problems"
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
struct LinesearchProblem{Tx, TF, TD} <: AbstractOptimizerProblem{Tx}
    F::TF
    D::TD
end

LinesearchProblem{Tx}(f, d) where {Tx <: Number} = LinesearchProblem{Tx, typeof(f), typeof(d)}(f, d)

LinesearchProblem(f, d, ::Tx=zero(Float64)) where {Tx <: Number} = LinesearchProblem{Tx}(f, d)

value(obj::LinesearchProblem, x::Number) = obj.F(x)
derivative(obj::LinesearchProblem, x::Number) = obj.D(x)

"""
    OptimizerProblem <: AbstractOptimizerProblem

Stores *gradients*. Also compare this to [`NonlinearProblem`](@ref).

The type of the *stored gradient* has to be a subtype of [`Gradient`](@ref).

# Functor

If `OptimizerProblem` is called on a single function, the gradient is generated with [`GradientAutodiff`](@ref).
"""
mutable struct OptimizerProblem{T, TF <: Callable, TG <: Union{Callable, Missing}, TH <: Union{Callable, Missing}} <: AbstractOptimizerProblem{T}
    F::TF
    G::TG
    H::TH
end

function OptimizerProblem(F::Callable, G::TG, H::Callable, ::Tx) where {T<:Number, Tx<:AbstractArray{T}, TG <: Union{Callable, Missing}}
    OptimizerProblem{T, typeof(F), TG, typeof(H)}(F, G, H)
end

function OptimizerProblem(F::Callable, G::TG, ::Tx) where {T<:Number, Tx<:AbstractArray{T}, TG <: Union{Callable, Missing}}
    OptimizerProblem{T, typeof(F), TG, Missing}(F, G, missing)
end

function OptimizerProblem(F::Callable, x::Tx; gradient = missing, hessian = missing) where {T<:Number, Tx<:AbstractArray{T}}
    ismissing(hessian) ? OptimizerProblem(F, gradient, x) : OptimizerProblem(F, gradient, hessian, x)
end

Gradient(obj::OptimizerProblem) = obj.G

function GradientFunction(prob::OptimizerProblem{T}) where {T}
    GradientFunction{T, typeof(prob.F), typeof(Gradient(prob))}(prob.F, Gradient(prob))
end

function GradientFunction(::OptimizerProblem{T, TF, Missing}) where {T, TF <: Callable}
    error("There is no gradient stored in this `OptimizerProblem`!")
end

function gradient(prob::OptimizerProblem, x::AbstractVector)
    grad = GradientFunction(prob)
    grad(x)
end