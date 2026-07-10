# As in `SufficientDecreaseCondition`, the value field carries a `₀` subscript
# (derivative at the base point α = 0) so it does not differ from the callable
# `D` only by letter case.
@doc raw"""
    CurvatureCondition <: BacktrackingCondition

The second of the Wolfe conditions [nocedal2006numerical](@cite). The first one is the [`SufficientDecreaseCondition`](@ref).

This encompasses the *standard curvature condition* and the *strong curvature condition*. This can be specified via the `mode` keyword.

With the standard curvature condition we check:
```math
f'(\alpha) ≥ c_2 d,
```
where ``c_2`` is the associated hyperparameter and ``d`` is the derivative at ``\alpha_0``. Further note that ``f'(\alpha_0)`` and ``d`` should both be negative.

With the strong curvature condition we check:
```math
|f'(\alpha)| ≤ c_2 |d|.
```

# Constructor

```julia
CurvatureCondition(c, d₀, D, Val(:Standard))
CurvatureCondition(c, d₀, D, Val(:Strong))
```
Here `D` has to be a function computing the derivative of the objective. The mode
is passed as a `Val` (defaulting to `Val(:Standard)`) so that it is encoded in the
type and dispatch — and hence inference — is stable without relying on constant
propagation of a `Symbol` keyword. The other inputs are numbers.
"""
struct CurvatureCondition{T,DT<:Callable,COND} <: BacktrackingCondition{T}
    c::T
    d₀::T

    D::DT

    function CurvatureCondition(c::T, d₀::T, D::DT, ::Val{COND}=Val(:Standard)) where {T<:Number,DT<:Callable,COND}
        @assert ((COND == :Standard) || (COND == :Strong)) "Mode has to be either :Strong or :Standard!"
        @assert zero(T) < c < one(T) "The curvature constant c must lie in (0, 1), it is $(c)."
        @assert !isnan(d₀) "d₀ is NaN"
        new{T,DT,COND}(c, d₀, D)
    end
end

function (cc::CurvatureCondition{T,DT,:Standard})(α::T) where {T,DT}
    cc.D(α) ≥ cc.c * cc.d₀
end

function (cc::CurvatureCondition{T,DT,:Strong})(α::T) where {T,DT}
    abs(cc.D(α)) ≤ abs(cc.c * cc.d₀)
end
