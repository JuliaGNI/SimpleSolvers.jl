"""
    BacktrackingCondition

Abstract type comprising the conditions that are used for checking step sizes for the backtracking line search (see [`BacktrackingState`](@ref)).
"""
abstract type BacktrackingCondition{T} where {T<:Number}
end

function compute_new_iterate(xₖ::VT, αₖ::T, ∇ₖf::TVT)::VT where {T, VT, TVT}
    error("Not implemented for $(VT).")
end

function compute_new_iterate(xₖ::VT, αₖ::T, ∇ₖf::TVT)::VT where {T, VT <: AbstractArray{T}, TVT <: AbstractArray{T}}
    xₖ + αₖ * ∇ₖf
end