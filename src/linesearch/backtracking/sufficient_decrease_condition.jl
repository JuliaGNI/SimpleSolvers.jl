# The value fields carry a `₀` subscript (value/derivative at the base point
# α = 0) so they do not differ from the callable `F` only by letter case
# (bugs.md §5: the former `f` vs `F` was an easy silent typo).
@doc raw"""
    SufficientDecreaseCondition <: BacktrackingCondition

The condition that determines if the change induced by ``\alpha_k`` is *big enough*. This is used in [`Backtracking`](@ref).

# Example

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: SufficientDecreaseCondition)
c = SimpleSolvers.DEFAULT_WOLFE_c₁
f(x) = (x - 1.) ^ 2
xₖ = 0.
fₖ = f(xₖ)
dₖ = 2xₖ - 2.

sdc = SufficientDecreaseCondition(c, fₖ, dₖ, f)
sdc(1.9), sdc(2.)

# output

(true, false)
```

# Extended help

We call the constant that pertains to the sufficient decrease condition ``c``. This is typically called ``c_1`` in the literature [nocedal2006numerical](@cite).
See [`DEFAULT_WOLFE_c₁`](@ref) for the relevant constant
"""
struct SufficientDecreaseCondition{T,FT} <: BacktrackingCondition{T}
    c::T
    f₀::T
    d₀::T

    F::FT

    function SufficientDecreaseCondition(c::Tc, f₀::T, d₀::T, F::FT) where {Tc<:Number,T<:Number,FT<:Callable}
        @assert T == Tc "You are computing with mixed precision ($(T) and $(Tc)). This is probably not intended (and not supported)."
        @assert !isnan(f₀) "f₀ is NaN"
        @assert !isnan(d₀) "d₀ is NaN"
        new{T,FT}(c, f₀, d₀, F)
    end
end

function (sdc::SufficientDecreaseCondition{T})(α::T) where {T}
    sdc.F(α) ≤ sdc.f₀ + sdc.c * α * sdc.d₀
end
