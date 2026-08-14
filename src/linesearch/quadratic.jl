"""
A factor by which `s` is reduced in each bracketing iteration (see [`bracket_minimum_with_fixed_point`](@ref)).
"""
const DEFAULT_s_REDUCTION = 0.5

@doc raw"""
    Quadratic <: LinesearchMethod

Quadratic Polynomial line search based on the polynomial
```math
p(α) = p_0 + p_1(\alpha - \alpha_0) + p_2(\alpha - \alpha_0)^2.
```
Performs multiple iterations in which all parameters ``p_0``, ``p_1`` and ``p_2`` are adapted.
We do not check the [`SufficientDecreaseCondition`](@ref) here. We instead repeatedly build new quadratic polynomials until a minimum is found (to sufficient accuracy).
The iteration may also stop after it reaches the maximum number of iterations, the
`linesearch_max_iterations` field of [`Options`](@ref) (see [`linesearch_iterations`](@ref)).

# Keywords

- `ε`: A constant that checks the *precision*/*tolerance*.
- `s`: A constant that determines the initial interval for bracketing. By default this is [`DEFAULT_BRACKETING_s`](@ref).
- `s_reduction:` A constant that determines the factor by which `s` is decreased in each new *bracketing iteration*.
- `αmax`: the largest step the bracketing will try, by default [`DEFAULT_LINESEARCH_αmax`](@ref).
  Without it the bracket grows outward until the merit stops falling, which for a nearly flat or
  distantly-minimised ``\varphi`` is arbitrarily far; see [`linesearch_αmax`](@ref), which is also
  how a caller imposes a *smaller* ceiling of its own.

# Extended help

The *quadratic* method. Compare this to [`BierlaireQuadratic`](@ref). The algorithm is adjusted from [kelley1995iterative](@cite).
"""
struct Quadratic{T} <: LinesearchMethod{T}
    ε::T
    s::T
    s_reduction::T
    αmax::T

    function Quadratic{T}(ε::T, s::T, s_reduction::T, αmax::T=T(DEFAULT_LINESEARCH_αmax)) where {T}
        @assert ε > 0 "Precision ε must be positive."
        @assert s > 0 "Bracketing step s must be positive."
        @assert 0 < s_reduction < 1 "Bracketing step reduction factor must satisfy 0 < s_reduction < 1."
        @assert αmax > 0 "The maximum step length must be positive, it is $(αmax)."
        new{T}(ε, s, s_reduction, αmax)
    end
end

function Quadratic(::Type{T}=Float64;
    ε=default_precision(T),
    s=T(DEFAULT_BRACKETING_s),
    s_reduction=T(DEFAULT_s_REDUCTION),
    αmax=T(DEFAULT_LINESEARCH_αmax)
) where {T}
    Quadratic{T}(ε, s, s_reduction, αmax)
end

method_αmax(m::Quadratic) = m.αmax

Quadratic(::Type{T}, ::SolverMethod) where {T} = Quadratic(T)

"""
    solve_with_status(ls::Linesearch{T,<:Quadratic}, α, params)

Fit successive quadratics to approximate the line minimiser and return the
[`LinesearchStatus`](@ref), emitting no messages. [`solve`](@ref) is this plus the report; see
[`Quadratic`](@ref).
"""
function solve_with_status(ls::Linesearch{T,<:Quadratic}, α₀::T, params=NullParameters()) where {T}
    # Before any merit evaluation, so that an unusable caller-supplied ceiling costs none.
    αmax = linesearch_αmax(method(ls), params)
    φ₀ = value(problem(ls), zero(T), params)
    d₀ = derivative(problem(ls), zero(T), params)

    anchor = check_anchor(φ₀, d₀, α₀, αmax)
    isnothing(anchor) || return anchor

    τ = armijo_tolerance(φ₀, armijo_ulps(T))
    # Every step this function can hand back is derived from the trial step or from the bracketing,
    # and both are bounded here rather than at each of the returns below.
    α₀ = min(α₀, αmax)
    αres, n = _quadratic_search(ls, α₀, params, αmax)

    # `bracket_minimum_with_fixed_point` flips direction when the merit rises to the right of
    # the bracketing *start* — which is α₀, not 0 — so even a decreasing anchor can yield a
    # bracket, and hence a minimiser, left of zero once α₀ overshoots. A negative step is not a
    # meaningful step length along a direction (see the α > 0 contract), so retry once from the
    # α = 0 anchor, which `check_anchor` has established is decreasing.
    if isnothing(αres) || αres ≤ zero(T)
        αres, nretry = _quadratic_search(ls, zero(T), params, αmax)
        n += nretry
    end
    # `bracket_minimum_with_fixed_point` fails only by exhausting `nmax` in both directions, i.e.
    # for a merit that keeps decreasing. That is a failure to *report*, not a round-off floor:
    # reporting a floor would make the outer iteration count a descending merit as stagnation.
    isnothing(αres) && return LinesearchStatus{T}(α₀, LINESEARCH_EXHAUSTED, n, φ₀, d₀, φ₀, τ, zero(T))
    # Still non-positive: no positive step improves the merit as far as this search can tell,
    # which is the floor — `check_anchor` established above that the anchor itself descends.
    αres > zero(T) || return LinesearchStatus{T}(α₀, LINESEARCH_FLOOR, n, φ₀, d₀, φ₀, τ, zero(T))

    # The bracketing already stops at the ceiling and the fit is confined to the bracket, so this
    # cannot bind; it is here so that the α ≤ αmax half of the contract is guaranteed by this
    # function rather than inferred from the two that feed it, and so that the merit reported
    # below is the merit at the step handed back whichever of them produced it.
    αres = min(αres, αmax)
    φres = value(problem(ls), αres, params)
    LinesearchStatus{T}(αres, φres ≤ φ₀ - τ ? LINESEARCH_DECREASED : LINESEARCH_FLOOR,
        n, φ₀, d₀, φres, τ, zero(T))
end

# The quadratic-fit iteration itself. Returns `(α, n)` with `n` the number of merit evaluations,
# or `(nothing, n)` if the merit cannot be bracketed.
# Private: `solve`/`solve_with_status` is the public entry point.
function _quadratic_search(ls::Linesearch{T,<:Quadratic}, α₀::T, params, αmax::T) where {T}
    n = 0
    # Start the bracketing at the caller's α₀ when it lies on the descent side
    # (φ′(α₀) < 0, so the minimiser is to its right); otherwise keep the α = 0 anchor,
    # where a descent direction is guaranteed decreasing. `bracket_minimum_with_fixed_point`
    # searches rightward from a fixed left point, so that point must be on the descent
    # side. See issue #164.
    α = (α₀ > zero(T) && derivative(problem(ls), α₀, params) < zero(T)) ? α₀ : zero(T)
    # A trial step at or beyond the ceiling leaves nothing to search: the whole admissible range
    # lies left of where the bracketing would start, so the ceiling itself is the answer.
    α < αmax || return (αmax, n)
    s = method(ls).s

    for _ in 1:config(ls).linesearch_max_iterations
        # fit p(α) = p₀ + p₁(α - a) + p₂(α - a)² with p₀ = y₀, p₁ = d₀ and
        # p₂ = (y₁ - y₀ - d₀(b - a)) / (b - a)²; the endpoint merits y₀, y₁ come
        # from the bracketing, so no re-evaluation is needed here.
        a, b, y₀, y₁, bracket = _bracket_minimum_with_fixed_point_core(problem(ls), params, α, s, T(DEFAULT_BRACKETING_k), DEFAULT_BRACKETING_nmax, αmax)
        # The merit could not be bracketed from here (see `bracket_minimum`).
        bracket === :unbracketable && return (nothing, n)
        # The bracket ends at the ceiling with the merit still falling across it, so the turning
        # point lies beyond the largest step the caller allows and `αmax` is the best admissible
        # step. Fitting the truncated bracket instead would be worse than useless: the fitted
        # curvature over a monotone-decreasing interval is non-positive, so the guard below falls
        # back to bisecting it and hands back a midpoint strictly above the endpoint's merit.
        bracket === :capped && return (αmax, n)
        n += 2   # this round of the fit; the bracketer’s own evaluations are not counted
        d₀ = derivative(problem(ls), a, params)
        # `d₀` is the derivative at the bracket's left endpoint `a`; return that point
        # (not the loop's start `α`), which differ when the bracketer flipped because
        # the start was not on the descent side.
        abs(d₀) < method(ls).ε && return (a, n)

        # minimizer αₜ = a - p₁ / (2p₂); guard on the fitted curvature (denom = 2p₂(b-a)²).
        # A non-positive curvature (denom ≤ 0), a non-finite αₜ, or a minimizer outside
        # the bracket means the quadratic model is untrustworthy — bisect [a, b] instead.
        denom = 2 * (y₁ - y₀ - d₀ * (b - a))
        αₜ = denom > zero(T) ? a - d₀ * (b - a)^2 / denom : (a + b) / 2
        (isfinite(αₜ) && a ≤ αₜ ≤ b) || (αₜ = (a + b) / 2)

        (l2norm(αₜ - α) < method(ls).ε) && return (αₜ, n)

        α = αₜ
        s *= method(ls).s_reduction
    end

    (α, n)
end

Base.show(io::IO, ls::Quadratic) = print(io, "Quadratic Polynomial with ε = $(ls.ε), s = $(ls.s), s_reduction = $(ls.s_reduction) and αmax = $(ls.αmax).")

function change_precision(::Type{T}, method::Quadratic) where {T}
    T ≠ eltype(method) || return method
    Quadratic{T}(T(method.ε), T(method.s), T(method.s_reduction), T(method.αmax))
end

function Base.isapprox(qu₁::Quadratic{T}, qu₂::Quadratic{T}; kwargs...) where {T}
    isapprox(qu₁.ε, qu₂.ε; kwargs...) && isapprox(qu₁.s, qu₂.s; kwargs...) && isapprox(qu₁.s_reduction, qu₂.s_reduction; kwargs...) && isapprox(qu₁.αmax, qu₂.αmax; kwargs...)
end
