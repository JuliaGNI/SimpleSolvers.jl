const MAX_NUMBER_ADJUST_CONSTANT_ITERATIONS = 5

"""
    triple_point_finder(f, x)

Find three points `a < b < c` (strictly ordered in position) with `f(a) ≥ f(b)` and `f(c) > f(b)`, so that a minimum is bracketed in `(a, c)`. This is used for performing a quadratic line search (see [`BierlaireQuadratic`](@ref)). Returns `nothing` when no such triple exists — see the warning below.

!!! note
    The left inequality is *non-strict* (`f(a) ≥ f(b)`): while descending, consecutive
    samples may tie on a plateau, and for a flat-bottomed `f` a strict `f(a) > f(b)`
    is unattainable. `f(b)` is still strictly below `f(c)`, and `BierlaireQuadratic`
    guards a degenerate (collinear) fit by falling back to a bisection step, so the
    non-strict left bound is sufficient to bracket the minimum.

!!! warning "Searches rightward only, and returns `nothing` on failure"
    Unlike [`bracket_minimum`](@ref), which *flips* direction when `f` increases to the right,
    `triple_point_finder` only ever searches in the direction of increasing `x` and therefore
    requires `f` to be decreasing at `x₀`. A caller that cannot guarantee that — a line search
    whose direction came from a stale or regularized [`Jacobian`](@ref), say — must check the
    anchor itself (see [`check_anchor`](@ref)).

    When no triple can be found the function returns `nothing` rather than raising: a line
    search must be able to *report* an unbracketable merit rather than abort the enclosing
    solve. Two situations produce it — `f` is not decreasing at `x₀` even after `δ` has been
    reduced, or `nmax` doublings did not reach a turning point.

# Implementation

For `δ` we take [`DEFAULT_BRACKETING_s`](@ref) as default. For `nmax` we take [`DEFAULT_BRACKETING_nmax`](@ref) as default.

# Examples

```jldoctest; setup = :(using SimpleSolvers: triple_point_finder; round10(x) = round(x; digits=10))
julia> f(x) = x ^ 2
f (generic function with 1 method)

julia> x = -1.
-1.0

julia> a, b, c = round10.(triple_point_finder(f, x))
(-0.37, 0.27, 1.55)

julia> round10.((f(a), f(b), f(c)))
(0.1369, 0.0729, 2.4025)
```

# Extended help

The algorithm is taken from [bierlaire2015optimization; Chapter 11.2.1](@cite).
"""
function triple_point_finder(f::Callable, x₀::T, δ::T, nmax::Integer=DEFAULT_BRACKETING_nmax, adjust_constant_iteration::Integer=1) where {T}
    fx₀ = f(x₀)
    x₁ = x₀ + δ
    fx₁ = f(x₁)

    if fx₁ ≥ fx₀
        # Halving δ is the right answer to an *overshoot* — the merit rose because the probe
        # stepped past a nearby minimum, so a shorter probe lands on the descending side. It is
        # the wrong answer when the rise is within round-off of `f(x₀)`: the merit then simply
        # does not resolve a decrease here, and a *smaller* δ is strictly less informative, so
        # the remaining halvings are wasted evaluations that end in the same failure. The
        # allowance matches `armijo_tolerance(fx₀, DEFAULT_ARMIJO_τ_ULPS)`, which is defined in
        # `src/linesearch/backtracking.jl` and so cannot be called from here without inverting
        # the `bracketing` → `linesearch` include order.
        τ = 4 * eps(fx₀)
        fx₁ ≤ fx₀ + τ && return nothing
        adjust_constant_iteration > MAX_NUMBER_ADJUST_CONSTANT_ITERATIONS && return nothing
        return triple_point_finder(f, x₀, δ / 2, nmax, adjust_constant_iteration + 1)
    end

    local xₖ₋₁ = x₀
    local xₖ = x₀
    local xₖ₊₁ = x₁
    local fxₖ = fx₀
    local fxₖ₊₁ = fx₁
    local increment = δ

    for k in 1:nmax
        xₖ₋₁ = xₖ
        xₖ = xₖ₊₁
        fxₖ = fxₖ₊₁
        increment = 2 * increment
        xₖ₊₁ = xₖ + increment
        fxₖ₊₁ = f(xₖ₊₁)
        if fxₖ₊₁ > fxₖ
            return (xₖ₋₁, xₖ, xₖ₊₁)
        end
    end

    nothing
end

function triple_point_finder(f::Callable, x₀::T; δ::T=T(DEFAULT_BRACKETING_s), nmax::Integer=DEFAULT_BRACKETING_nmax, adjust_constant_iteration::Integer=1) where {T}
    triple_point_finder(f, x₀, δ, nmax, adjust_constant_iteration)
end

function triple_point_finder(prob::LinesearchProblem{T}, params, x₀::T; δ::T=T(DEFAULT_BRACKETING_s), nmax::Integer=DEFAULT_BRACKETING_nmax) where {T}
    triple_point_finder(x -> value(prob, x, params), x₀, δ, nmax)
end
