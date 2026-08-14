const MAX_NUMBER_ADJUST_CONSTANT_ITERATIONS = 5

"""
    triple_point_finder(f, x)

Find three points `a < b < c` (strictly ordered in position) with `f(a) ≥ f(b)` and `f(c) > f(b)`, so that a minimum is bracketed in `(a, c)`. This is used for performing a quadratic line search (see [`BierlaireQuadratic`](@ref)). Returns a `Symbol` instead of a triple when no such triple exists — see the warning below.

!!! note
    The left inequality is *non-strict* (`f(a) ≥ f(b)`): while descending, consecutive
    samples may tie on a plateau, and for a flat-bottomed `f` a strict `f(a) > f(b)`
    is unattainable. `f(b)` is still strictly below `f(c)`, and `BierlaireQuadratic`
    guards a degenerate (collinear) fit by falling back to a bisection step, so the
    non-strict left bound is sufficient to bracket the minimum.

!!! warning "Searches rightward only, and reports failure as a `Symbol`"
    Unlike [`bracket_minimum`](@ref), which *flips* direction when `f` increases to the right,
    `triple_point_finder` only ever searches in the direction of increasing `x` and therefore
    requires `f` to be decreasing at `x₀`. A caller that cannot guarantee that — a line search
    whose direction came from a stale or regularized [`Jacobian`](@ref), say — must check the
    anchor itself (see [`check_anchor`](@ref)).

    When no triple can be found the function returns a `Symbol` rather than raising: a line
    search must be able to *report* an unbracketable merit rather than abort the enclosing
    solve. The two failures mean opposite things and are therefore distinguished, because a
    caller that conflates them reports a *descending* merit as stagnation:

    - `:flat` — the rise at the first probe is within the round-off resolution of `f(x₀)`
      ([`armijo_tolerance`](@ref)), so `f` does not resolve a decrease here at all. No line
      search can improve on this point (`LINESEARCH_FLOOR`).
    - `:unbracketable` — there *is* a decrease, but it cannot be bracketed: either `nmax`
      doublings never reached a turning point, or `f` rose at every probe down to the smallest
      `δ` tried. This is a genuine failure to report (`LINESEARCH_EXHAUSTED`), not a floor.

    A third status, `:capped`, is not a failure at all: the search reached the ceiling `αmax` (see
    [`linesearch_αmax`](@ref)) while `f` was still falling, so the turning point lies beyond the
    largest step the caller allows and that ceiling *is* the answer. It is reachable only through
    `_triple_point_core`, which is where the `αmax` argument lives; `triple_point_finder` itself
    takes no ceiling and therefore never reports it.

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
    a, b, c, _, status = _triple_point_core(f, x₀, δ, nmax, adjust_constant_iteration)
    status === :ok ? (a, b, c) : status
end

function triple_point_finder(f::Callable, x₀::T; δ::T=T(DEFAULT_BRACKETING_s), nmax::Integer=DEFAULT_BRACKETING_nmax, adjust_constant_iteration::Integer=1) where {T}
    triple_point_finder(f, x₀, δ, nmax, adjust_constant_iteration)
end

function triple_point_finder(prob::LinesearchProblem{T}, params, x₀::T; δ::T=T(DEFAULT_BRACKETING_s), nmax::Integer=DEFAULT_BRACKETING_nmax) where {T}
    a, b, c, _, status = _triple_point_core(prob, params, x₀; δ=δ, nmax=nmax)
    status === :ok ? (a, b, c) : status
end

# The bracketing loop, returning `(a, b, c, n, status)` with `n` the number of evaluations of `f`
# it spent, `status` one of `:ok`, `:flat`, `:capped` or `:unbracketable`, and the three points
# meaningful only for `:ok`. `n` is what lets `BierlaireQuadratic` report the true cost of a search
# whose bracketing *is* the search, which is what a binding ceiling produces.
#
# Splitting the loop from the reporting is what the `Bisection` line search does with
# `bisection`/`_bisection_core`, and here it
# also keeps the return type concrete: the `Union{Symbol,Tuple}` handed back by
# `triple_point_finder` has to be boxed, and `BierlaireQuadratic` brackets once per line search.
function _triple_point_core(f::Callable, x₀::T, δ::T, nmax::Integer, adjust_constant_iteration::Integer, αmax::T=T(Inf)) where {T}
    # A ceiling at or below the start leaves no admissible range to search at all. Without this
    # the clamp below would make `δ` zero or negative, every probe would land at or *left* of
    # `x₀`, and the rise test would fire on the start itself — reporting `:flat`, i.e. that no
    # line search can decrease the merit here, about a search that was never allowed to look.
    # The ceiling is the answer instead, and saying so costs no evaluation. `BierlaireQuadratic`
    # guards this before it calls, but the guard belongs here, where the `αmax` argument is.
    αmax > x₀ || return (x₀, x₀, x₀, 0, :capped)
    fx₀ = f(x₀)
    n = 1
    # The first probe has to stay inside the ceiling too, or a caller whose ceiling is smaller
    # than the default `δ` would have its whole search happen outside the range it allows. Only
    # a rightward `δ` is clamped: a negative one is not a step length and the guard above has
    # already established that there is room to the right.
    δ = min(δ, αmax - x₀)
    x₁ = x₀ + δ
    fx₁ = f(x₁)
    n += 1

    if fx₁ ≥ fx₀
        # Halving δ is the right answer to an *overshoot* — the merit rose because the probe
        # stepped past a nearby minimum, so a shorter probe lands on the descending side. It is
        # the wrong answer when the rise is within round-off of `f(x₀)`: the merit then simply
        # does not resolve a decrease here, and a *smaller* δ is strictly less informative, so
        # the remaining halvings are wasted evaluations that end in the same failure.
        fx₁ ≤ fx₀ + armijo_tolerance(fx₀, armijo_ulps(typeof(fx₀))) && return (x₀, x₀, x₁, n, :flat)
        adjust_constant_iteration > MAX_NUMBER_ADJUST_CONSTANT_ITERATIONS && return (x₀, x₀, x₁, n, :unbracketable)
        # The halvings are rounds of this same search, so their evaluations are this search's too.
        a, b, c, nretry, status = _triple_point_core(f, x₀, δ / 2, nmax, adjust_constant_iteration + 1, αmax)
        return (a, b, c, n + nretry, status)
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
        # Probe *at* the ceiling rather than past it, and stop there: `f` was still falling at
        # `xₖ`, so a turning point (if any) lies beyond the largest step the caller allows. The
        # rise can still happen exactly at the ceiling, which is a genuine triple, hence the test
        # — with `xₖ₊₁ > xₖ` alongside it, since a ceiling at or below `xₖ` cannot order one.
        if xₖ₊₁ ≥ αmax
            xₖ₊₁ = αmax
            # A ceiling at or below `xₖ` cannot order a triple, and when it is *equal* to `xₖ` —
            # which it is whenever the first probe was clamped to it — `fxₖ` already is the merit
            # there, so evaluating again would only pay for a value in hand.
            xₖ₊₁ > xₖ || return (xₖ₋₁, xₖ, xₖ₊₁, n, :capped)
            fxₖ₊₁ = f(xₖ₊₁)
            n += 1
            return (xₖ₋₁, xₖ, xₖ₊₁, n, fxₖ₊₁ > fxₖ ? :ok : :capped)
        end
        fxₖ₊₁ = f(xₖ₊₁)
        n += 1
        if fxₖ₊₁ > fxₖ
            return (xₖ₋₁, xₖ, xₖ₊₁, n, :ok)
        end
    end

    # `nmax` doublings without a turning point: the merit is still descending at
    # `x₀ + δ(2^(nmax+1) - 1)`. That is the opposite of `:flat` — there is a decrease here, it
    # just cannot be bracketed — so the caller must not report it as a round-off floor.
    (xₖ₋₁, xₖ, xₖ₊₁, n, :unbracketable)
end

function _triple_point_core(prob::LinesearchProblem{T}, params, x₀::T; δ::T=T(DEFAULT_BRACKETING_s), nmax::Integer=DEFAULT_BRACKETING_nmax, αmax::T=T(Inf)) where {T}
    _triple_point_core(x -> value(prob, x, params), x₀, δ, nmax, 1, αmax)
end
