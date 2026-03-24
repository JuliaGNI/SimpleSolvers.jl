@doc raw"""
    CurvatureCondition <: LinesearchCondition

The second of the Wolfe conditions [nocedal2006numerical](@cite). The first one is the [`SufficientDecreaseCondition`](@ref).

This encompasses the *standard curvature condition* and the *strong curvature condition*. This can be specified via the `mode` keyword.

With the standard curvature condition we check:
```math
f'(\alpha) ≥ c_2 d,
```
where ``c_2`` is the associated hyperparameter and ``d`` is the derivative at ``\alpha_0``. Further note that ``f'(\alpha_0)`` and ``d`` should both be negative.

With the strong curvature condition we check:
```math
|f'(\alpha)| < c_2 |d|.
```

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
