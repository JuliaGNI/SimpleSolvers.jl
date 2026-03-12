@doc raw"""
    SufficientDecreaseCondition <: LinesearchCondition

The condition that determines if ``\alpha_k`` is *big enough*.

# Constructor

```julia
SufficientDecreaseCondition(c, xₖ, fₖ, dₖ, pₖ, f)
```

# Functors

```julia
sdc(αₖ₊₁, αₖ)
sdc(αₖ)
```
The second functor is shorthand for `sdc(compute_new_iterate(sdc.αₖ, αₖ, sdc.pₖ), T(αₖ))`, also see [`compute_new_iterate!`](@ref).

# Extended help

We call the constant that pertains to the sufficient decrease condition ``c``. This is typically called ``c_1`` in the literature [nocedal2006numerical](@cite).
See [`DEFAULT_WOLFE_c₁`](@ref) for the relevant constant
"""
struct SufficientDecreaseCondition{T,FT} <: BacktrackingCondition{T}
    c::T
    f::T
    d::T

    F::FT

    function SufficientDecreaseCondition(c::Tc, f::T, d::T, F::FT) where {Tc<:Number,T<:Number,FT<:Callable}
        @assert T == Tc "You are computing with mixed precision ($(T) and $(Tc)). This is probably not intended (and not supported)."
        @assert !isnan(f) "f is NaN"
        @assert !isnan(d) "d is NaN"
        new{T,FT}(c, f, d, F)
    end
end

function (sdc::SufficientDecreaseCondition{T})(α::T) where {T}
    sdc.F(α) ≤ sdc.f + sdc.c * α * sdc.d
end
