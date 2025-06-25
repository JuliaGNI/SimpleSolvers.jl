@doc raw"""
    CurvatureCondition <: LinesearchCondition

The second of the Wolfe conditions [nocedal2006numerical](@cite). The first one is the [`SufficientDecreaseCondition`](@ref).

This encompasses the *standard curvature condition* and the *strong curvature condition*.

# Constructor

```julia
CurvatureCondition(c, xₖ, gradₖ, pₖ, obj, grad; mode)
```
Here `grad` has to be a [`Gradient`](@ref) and `obj` an [`AbstractObjective`](@ref). The other inputs are either arrays or numbers.

# Implementation

For computational reasons `CurvatureCondition` also has a field `gradₖ₊₁` in which the temporary new gradient is saved.
"""
mutable struct CurvatureCondition{T, VT <: Union{T, AbstractArray{T}}, TVT <: Union{T, AbstractArray{T}}, OT <: AbstractObjective{T}, GT <: Union{Callable, Gradient{T}}, CCT} <: BacktrackingCondition{T}
    c::T
    xₖ::VT
    gradₖ::TVT
    pₖ::TVT
    obj::OT
    grad::GT
    gradₖ₊₁::TVT
    function CurvatureCondition(c::T, xₖ::VT, gradₖ::TVT, pₖ::TVT, obj::OT, grad::GT; mode=:Standard) where {T <: Number, VT <: AbstractArray{T}, TVT <: AbstractArray{T}, OT <: MultivariateObjective{T}, GT <: Gradient{T}}
        @assert ((mode == :Standard) || (mode == :Strong)) "Mode has to be either :Strong or :Standard!"
        new{T, VT, TVT, OT, GT, mode}(c, xₖ, gradₖ, pₖ, obj, grad, alloc_g(xₖ))
    end
    function CurvatureCondition(c::T, xₖ::T, dₖ::T, pₖ::T, obj::OT, d::DT; mode=:Standard) where {T <: Number, DT, OT <: AbstractUnivariateObjective}
        @assert ((mode == :Standard) || (mode == :Strong)) "Mode has to be either :Strong or :Standard!"
        new{T, T, T, OT, DT, mode}(c, xₖ, dₖ, pₖ, obj, d, alloc_d(xₖ))
    end
end

function standard_curvature_condition(cc::CurvatureCondition{T, VT, TVT, OT, GT}, xₖ₊₁::VT, αₖ::T) where {T, VT, TVT, OT, GT}
    gradient!(cc.gradₖ₊₁, xₖ₊₁, cc.grad)' * cc.pₖ ≥ cc.c * cc.gradₖ' * cc.pₖ
end

function strong_curvature_condition(cc::CurvatureCondition{T, VT, TVT, OT, GT}, xₖ₊₁::VT, αₖ::T) where {T, VT, TVT, OT, GT}
    abs(gradient!(cc.gradₖ₊₁, xₖ₊₁, cc.grad)' * cc.pₖ) < abs(cc.c * cc.gradₖ' * cc.pₖ)
end

function standard_curvature_condition(cc::CurvatureCondition{T, T, T, OT, GT}, xₖ₊₁::T, αₖ::T) where {T, OT, GT}
    __derivative!(cc.obj, xₖ₊₁)' * cc.pₖ ≥ cc.c * cc.gradₖ' * cc.pₖ
end

function strong_curvature_condition(cc::CurvatureCondition{T, T, T, OT, GT}, xₖ₊₁::T, αₖ::T) where {T, OT, GT}
    abs(__derivative!(cc.obj, xₖ₊₁)' * cc.pₖ) < abs(cc.c * cc.gradₖ' * cc.pₖ)
end

function (cc::CurvatureCondition{T, VT, TVT, OT, GT, :Standard})(xₖ₊₁::VT, αₖ::T) where {T, VT, TVT, OT, GT}
    standard_curvature_condition(cc, xₖ₊₁, αₖ)
end

function (cc::CurvatureCondition{T, VT, TVT, OT, GT, :Strong})(xₖ₊₁::VT, αₖ::T) where {T, VT, TVT, OT, GT}
    strong_curvature_condition(cc, xₖ₊₁, αₖ)
end

function (cc::CurvatureCondition{T})(αₖ::T) where {T}
    cc(compute_new_iterate(cc.xₖ, αₖ, cc.pₖ), αₖ)
end