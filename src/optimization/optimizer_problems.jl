"""
    AbstractOptimizerProblem

An *optimizer problem* is a quantity to has to be made zero by a solver or minimized by an optimizer.

See [`LinesearchProblem`](@ref) and [`OptimizerProblem`](@ref).
"""
abstract type AbstractOptimizerProblem{T<:Number} <: AbstractProblem end


"""
    value(obj::AbstractOptimizerProblem, x)

Evaluates the value at `x` (i.e. computes `obj.F(x)`).
"""
function value(obj::AbstractOptimizerProblem, x::Union{Number,AbstractArray{<:Number}})
    obj.F(x)
end


"""
    OptimizerProblem <: AbstractOptimizerProblem

Stores *gradients*. Also compare this to [`NonlinearProblem`](@ref).

The type of the *stored gradient* has to be a subtype of [`Gradient`](@ref).

# Functor

If `OptimizerProblem` is called on a single function, the gradient is generated with [`GradientAutodiff`](@ref).
"""
mutable struct OptimizerProblem{T,TF<:Callable,TG<:Union{Callable,Missing},TH<:Union{Callable,Missing}} <: AbstractOptimizerProblem{T}
    F::TF
    G::TG
    H::TH
end

function OptimizerProblem(F::Callable, G::TG, H::Callable, ::Tx) where {T<:Number,Tx<:AbstractArray{T},TG<:Union{Callable,Missing}}
    OptimizerProblem{T,typeof(F),TG,typeof(H)}(F, G, H)
end

function OptimizerProblem(F::Callable, G::TG, ::Tx) where {T<:Number,Tx<:AbstractArray{T},TG<:Union{Callable,Missing}}
    OptimizerProblem{T,typeof(F),TG,Missing}(F, G, missing)
end

function OptimizerProblem(F::Callable, x::Tx; gradient=missing, hessian=missing) where {T<:Number,Tx<:AbstractArray{T}}
    ismissing(hessian) ? OptimizerProblem(F, gradient, x) : OptimizerProblem(F, gradient, hessian, x)
end

gradient(obj::OptimizerProblem) = obj.G
hessian(obj::OptimizerProblem) = obj.H

function GradientFunction(prob::OptimizerProblem{T}) where {T}
    GradientFunction{T,typeof(prob.F),typeof(Gradient(prob))}(prob.F, Gradient(prob))
end

function GradientFunction(::OptimizerProblem{T,TF,Missing}) where {T,TF<:Callable}
    error("There is no gradient stored in this `OptimizerProblem`!")
end

function gradient(prob::OptimizerProblem, x::AbstractVector)
    grad = GradientFunction(prob)
    grad(x)
end
