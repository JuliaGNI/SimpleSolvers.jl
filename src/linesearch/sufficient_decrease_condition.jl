# The value fields carry a `₀` subscript (value/derivative at the base point
# α = 0) so they do not differ from the callable `F` only by letter case
# (a former `f` vs `F` naming was an easy silent typo).
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

The optional keyword `τ` slackens the condition by an absolute amount:
```math
f(\alpha) \leq f_0 + c\alpha{}d_0 + \tau.
```
It defaults to zero, i.e. the exact condition. Without it the accept/reject decision is
taken by *rounding alone* as soon as ``c\alpha|d_0|`` drops below one unit in the last place
of ``f_0``: the right-hand side then rounds back up to ``f_0`` and the test degenerates to
``f(\alpha) \leq f_0``, which a merit that has reached its round-off floor passes or fails at
random. See [`armijo_tolerance`](@ref) and [`backtracking_αmin`](@ref) for how
[`Backtracking`](@ref) chooses ``\tau`` and derives a *meaningful* smallest step from it.
"""
struct SufficientDecreaseCondition{T,FT} <: BacktrackingCondition{T}
    c::T
    f₀::T
    d₀::T
    τ::T

    F::FT

    function SufficientDecreaseCondition(c::Tc, f₀::T, d₀::T, F::FT; τ::T=zero(T)) where {Tc<:Number,T<:Number,FT<:Callable}
        @assert T == Tc "You are computing with mixed precision ($(T) and $(Tc)). This is probably not intended (and not supported)."
        @assert !isnan(f₀) "f₀ is NaN"
        @assert !isnan(d₀) "d₀ is NaN"
        @assert τ ≥ zero(T) "The round-off allowance τ has to be nonnegative, it is $(τ)."
        new{T,FT}(c, f₀, d₀, τ, F)
    end
end

# The two-argument form takes a merit value `fα = F(α)` that the caller has already
# computed, so a backtracking loop that needs `f(α)` for its interpolation model does not
# pay for a second (possibly very expensive) evaluation of the merit.
function (sdc::SufficientDecreaseCondition{T})(α::T, fα::T) where {T}
    fα ≤ sdc.f₀ + sdc.c * α * sdc.d₀ + sdc.τ
end

function (sdc::SufficientDecreaseCondition{T})(α::T) where {T}
    sdc(α, sdc.F(α))
end
