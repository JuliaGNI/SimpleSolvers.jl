@doc raw"""
    CurvatureCondition <: LinesearchCondition

The second of the Wolfe conditions [nocedal2006numerical](@cite). The first one is the [`SufficientDecreaseCondition`](@ref).

This encompasses the *standard curvature condition* and the *strong curvature condition*.

# Constructor

```julia
CurvatureCondition(c, xₖ, dₖ, pₖ, grad; mode)
```
Here `grad` has to be a function computing the derivative of the objective. The other inputs are numbers.
"""
struct CurvatureCondition{T,DT<:Callable,COND} <: BacktrackingCondition{T}
    c::T
    xₖ::T
    dₖ::T
    pₖ::T

    d::DT

    function CurvatureCondition(c::T, xₖ::T, dₖ::T, pₖ::T, d::DT; mode=:Standard) where {T<:Number,DT}
        @assert ((mode == :Standard) || (mode == :Strong)) "Mode has to be either :Strong or :Standard!"
        @assert !isnan(xₖ) "xₖ is NaN"
        @assert !isnan(dₖ) "dₖ is NaN"
        @assert !isnan(pₖ) "pₖ is NaN"
        new{T,DT,mode}(c, xₖ, dₖ, pₖ, d)
    end
end

function (cc::CurvatureCondition{T,DT,:Standard})(xₖ₊₁::T, αₖ::T) where {T,DT}
    cc.d(xₖ₊₁) * cc.pₖ ≥ cc.c * cc.dₖ * cc.pₖ
end

function (cc::CurvatureCondition{T,DT,:Strong})(xₖ₊₁::T, αₖ::T) where {T,DT}
    abs(cc.d(xₖ₊₁) * cc.pₖ) < abs(cc.c * cc.dₖ * cc.pₖ)
end

function (cc::CurvatureCondition{T})(αₖ::T) where {T}
    cc(compute_new_iterate(cc.xₖ, αₖ, cc.pₖ), αₖ)
end
