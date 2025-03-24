@doc raw"""
    CurvatureCondition <: LinesearchCondition

The second of the Wolfe conditions [nocedal2006numerical](@cite). 

This encompasses the *standard curvature condition* and the *strong curvature condition*.
"""
mutable struct CurvatureCondition{T, VT <: AbstractArray{T}, TVT <: AbstractArray{T}, OT <: AbstractObjective, GT <: Gradient{T}, CCT} <: BacktrackingCondition
    c₂::T
    xₖ::VT
    gradₖ::TVT
    pₖ::TVT
    obj::OT
    grad::GT
end

function standard_curvature_condition(cc::CurvatureCondition{T, VT, TVT, OT, GT}, xₖ₊₁::VT, αₖ::T) where {T, VT, TVT, OT, GT}
    cc.grad(xₖ₊₁)' * cc.pₖ ≥ c₂ * cc.gradₖ' * cc.pₖ
end

function strong_curvature_condition(cc::CurvatureCondition{T, VT, TVT, OT, GT}, xₖ₊₁::VT, αₖ::T) where {T, VT, TVT, OT, GT}
    abs(gradient(cc.grad, xₖ₊₁)' * cc.pₖ) < abs(c₂ * cc.gradₖ' * cc.pₖ)
end

function (cc::CurvatureCondition{T, VT, TVT, OT, GT, :Standard})(xₖ₊₁::VT, αₖ::T) where {T, VT, TVT, OT, GT}
    standard_curvature_condition(cc, xₖ₊₁, αₖ)
end

function (cc::CurvatureCondition{T, VT, TVT, OT, GT, :Strong})(xₖ₊₁::VT, αₖ::T) where {T, VT, TVT, OT, GT}
    strong_curvature_condition(cc, xₖ₊₁, αₖ)
end

function (bc::CurvatureCondition{T})(αₖ::T) where {T}
    bc(compute_new_iterate(bc.xₖ, αₖ, bc.gradₖ), αₖ)
end