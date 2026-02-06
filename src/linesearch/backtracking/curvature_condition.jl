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
    d::T

    D::DT

    function CurvatureCondition(c::T, d::T, D::DT; mode=:Standard) where {T<:Number,DT<:Callable}
        @assert ((mode == :Standard) || (mode == :Strong)) "Mode has to be either :Strong or :Standard!"
        @assert !isnan(d) "d is NaN"
        new{T,DT,mode}(c, d, D)
    end
end

function (cc::CurvatureCondition{T,DT,:Standard})(α::T) where {T,DT}
    cc.D(α) ≥ cc.c * cc.d
end

function (cc::CurvatureCondition{T,DT,:Strong})(α::T) where {T,DT}
    abs(cc.D(α)) < abs(cc.c * cc.d)
end
