@doc raw"""
    SufficientDecreaseCondition <: LinesearchCondition

The condition that determines if ``\alpha_k`` is *big enough*.

# Constructor

```julia
SufficientDecreaseCondition(c, xₖ, fₖ, dₖ, pₖ, f)
```

# Functors

```julia
sdc(xₖ₊₁, αₖ)
sdc(αₖ)
```
The second functor is shorthand for `sdc(compute_new_iterate(sdc.xₖ, αₖ, sdc.pₖ), T(αₖ))`, also see [`compute_new_iterate!`](@ref).

# Extended help

We call the constant that pertains to the sufficient decrease condition ``c``. This is typically called ``c_1`` in the literature [nocedal2006numerical](@cite).
See [`DEFAULT_WOLFE_c₁`](@ref) for the relevant constant
"""
struct SufficientDecreaseCondition{T,FT} <: BacktrackingCondition{T}
    c::T
    xₖ::T
    fₖ::T
    dₖ::T
    pₖ::T

    f::FT

    function SufficientDecreaseCondition(c::Tc, xₖ::T, fₖ::T, dₖ::T, pₖ::T, f::FT) where {Tc<:Number,T<:Number,FT<:Callable}
        @assert T == Tc "You are computing with mixed precision ($(T) and $(Tc)). This is probably not intended (and not supported)."
        @assert !isnan(xₖ) "xₖ is NaN"
        @assert !isnan(fₖ) "fₖ is NaN"
        @assert !isnan(dₖ) "dₖ is NaN"
        @assert !isnan(pₖ) "pₖ is NaN"
        new{T,FT}(c, xₖ, fₖ, dₖ, pₖ, f)
    end
end

function (sdc::SufficientDecreaseCondition{T})(xₖ₊₁::T, αₖ::T) where {T}
    sdc.f(xₖ₊₁) ≤ sdc.fₖ + sdc.c * αₖ * sdc.pₖ * sdc.dₖ
end

function (sdc::SufficientDecreaseCondition{T})(αₖ::Tα) where {T,Tα}
    @assert T == Tα "You are computing with mixed precision ($(T) and $(Tα)). This is probably not intended (and not supported)."
    sdc(compute_new_iterate(sdc.xₖ, αₖ, sdc.pₖ), αₖ)
end
