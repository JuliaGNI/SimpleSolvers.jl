"""
    triple_point_finder(f, x)

Find three points `a > b > c` s.t. `f(a) > f(b)` and `f(c) > f(b)`. This is used for performing a quadratic line search (see [`QuadraticState`](@ref)).

# Implementation

For `δ` we take [`DEFAULT_BRACKETING_s`](@ref) as default. For `nmax we take [`DEFAULT_BRACKETING_nmax`](@ref) as default.

# Extended help

The algorithm is taken from [bierlaire2015optimization; Chapter 11.2.1](@cite).
"""
function triple_point_finder(f::Callable, x₀::T=0.0; δ::T=T(DEFAULT_BRACKETING_s), nmax::Integer=DEFAULT_BRACKETING_nmax) where {T}
    x₁ = x₀ + δ
    @assert f(x₁) < f(x₀) "The function `f` must be decreasing at `$(x₀)``; `f($(x₁)) = $(f(x₁))` must be smaller than `f($(x₀)) = $(f(x₀))`."
    local xₖ₋₁ = x₀
    local xₖ = x₁
    local xₖ₊₁ = xₖ
    local increment = δ
    for k in 1:nmax
        xₖ₋₁ = xₖ
        xₖ = xₖ₊₁
        increment = 2 * increment
        xₖ₊₁ = xₖ + increment
        if f(xₖ₊₁) > f(xₖ)
            return (xₖ₋₁, xₖ, xₖ₊₁)
        end
    end
    error("Unable to find a triple point for quadratic line search starting at x = $x₀.")
end

function triple_point_finder(obj::AbstractUnivariateProblem{T}, x₀::T=T(0.0); δ::T=T(DEFAULT_BRACKETING_s), nmax::Integer=DEFAULT_BRACKETING_nmax) where {T}
    triple_point_finder(obj.F, x₀; δ = δ, nmax = nmax)
end