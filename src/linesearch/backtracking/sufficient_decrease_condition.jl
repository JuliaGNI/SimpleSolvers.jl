@doc raw"""
    SufficientDecreaseCondition <: LinesearchCondition

The condition that determines if ``\alpha_k`` is *big enough*.
"""
mutable struct SufficientDecreaseCondition{T, VT<:Union{T, AbstractArray{T}}, TVT<:Union{T, AbstractArray{T}}, OT <:AbstractObjective} <: BacktrackingCondition
    c₁::T
    xₖ::VT
    fₖ::T
    gradₖ::TVT
    pₖ::TVT
    obj::OT

    # SufficientDecreaseCondition(c₁::T, xₖ::VT, fₖ::T, gradₖ::TVT, pₖ::TVT, obj::OT) where {T, VT, TVT, OT} = new{T, VT, TVT, OT}(c₁, xₖ, fₖ, gradₖ, pₖ, obj)
    function SufficientDecreaseCondition(c₁::T₁, xₖ::VT, fₖ::T, gradₖ::TVT, pₖ::TVT, obj::OT) where {T₁<:Number,
                                                                                                    T<:Number,
                                                                                                    VT <:Union{T₁, AbstractArray{T₁}},
                                                                                                    TVT<:Union{T, AbstractArray{T}},
                                                                                                    OT <: AbstractObjective}
    T != T₁ ? (@warn "You are computing with mixed precision ($(T) and $(T₁)). This is probably not intended.") : nothing
    xₖ_transformed = T.(xₖ)
    new{T, typeof(xₖ_transformed), TVT, OT}(T(c₁), xₖ_transformed, fₖ, gradₖ, pₖ, obj)
    end
end

function (sdc::SufficientDecreaseCondition{T, VT})(xₖ₊₁::VT, αₖ::T) where {T, VT}
    value!(sdc.obj, xₖ₊₁) < sdc.fₖ + sdc.c₁ * αₖ * sdc.pₖ' * sdc.gradₖ
end

function (bc::SufficientDecreaseCondition{T})(αₖ::T₁) where {T, T₁ <: Number}
    T != T₁ ? (@warn "You are computing with mixed precision ($(T) and $(T₁)). This is probably not intended.") : nothing
    bc(compute_new_iterate(bc.xₖ, αₖ, bc.gradₖ), T(αₖ))
end