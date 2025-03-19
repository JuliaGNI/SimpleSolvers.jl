"""
    SufficientDecreaseCondition <: LinesearchCondition

The condition that determines if ``\alpha_k`` is *big enough*.
"""
mutable struct SufficientDecreaseCondition{T, VT<:AbstractArray{T}, TVT<:AbstractArray{T}, OT <:AbstractObjective} <: LinesearchCondition{T}
    c₁::T
    xₖ::VT
    fₖ::T
    gradₖ::TVT
    pₖ::TVT
    obj::OT
end

function (sdc::SufficientDecreaseCondition{T, VT})(xₖ₊₁::VT, αₖ::T) where {T, VT}
    value(sdc.obj, xₖ₊₁) < sdc.fₖ + sdc.c₁ * αₖ * sdc.pₖ' * sdc.gradₖ
end