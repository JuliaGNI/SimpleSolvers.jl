"""
    BacktrackingCondition

Abstract type comprising the conditions that are used for checking step sizes for the backtracking line search (see [`BacktrackingState`](@ref)).
"""
abstract type BacktrackingCondition{T} end

(bc::BCT, xₖ, αₖ, gradₖ) where {BCT <: BacktrackingCondition} = error("Condition $(BCT) not defined for this combination of input arguments.")

# this is there because we also may have manifolds
function compute_new_iterate(xₖ::VT, αₖ::T, pₖ::TVT) where {T, VT, TVT}
    error("Not implemented for $(VT).")
end

function compute_new_iterate(xₖ::VT, αₖ::T, pₖ::TVT) where {T <: Number, VT <: Union{T, AbstractArray{T}}, TVT <: Union{T, AbstractArray{T}}}
    xₖ + αₖ * pₖ
end

function compute_new_iterate(xₖ::VT, αₖ::T₁, pₖ::TVT) where {T₁ <: Number, T <: Number, VT <: Union{T, AbstractArray{T}}, TVT <: Union{T, AbstractArray{T}}}
    @warn "Your are computing with mixed precisions ($(T₁) and $(T)). This is probably not on purpose."
    compute_new_iterate(xₖ, T(αₖ), pₖ)
end