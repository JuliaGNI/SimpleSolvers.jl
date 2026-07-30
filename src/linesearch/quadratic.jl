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

# Extended help

The *quadratic* method. Compare this to [`BierlaireQuadratic`](@ref). The algorithm is adjusted from [kelley1995iterative](@cite).
"""
struct Quadratic{T} <: LinesearchMethod{T}
    ε::T
    s::T
    s_reduction::T

    function Quadratic{T}(ε::T, s::T, s_reduction::T) where {T}
        @assert ε > 0 "Precision ε must be positive."
        @assert s > 0 "Bracketing step s must be positive."
        @assert 0 < s_reduction < 1 "Bracketing step reduction factor must satisfy 0 < s_reduction < 1."
        new{T}(ε, s, s_reduction)
    end
end

function Quadratic(::Type{T}=Float64;
    ε=default_precision(T),
    s=T(DEFAULT_BRACKETING_s),
    s_reduction=T(DEFAULT_s_REDUCTION)
) where {T}
    Quadratic{T}(ε, s, s_reduction)
end

Quadratic(::Type{T}, ::SolverMethod) where {T} = Quadratic(T)

"""
    solve(ls::Linesearch{T,<:Quadratic}, α, params)

Fit successive quadratics to approximate the line minimiser, report the outcome through
[`linesearch_warnings`](@ref) and return the step length. See [`Quadratic`](@ref) and
[`solve_with_status`](@ref).
"""
function solve(ls::Linesearch{T,<:Quadratic}, α::T, params=NullParameters()) where {T}
    status = solve_with_status(ls, α, params)
    linesearch_warnings(status, ls, params)
    steplength(status)
end

function solve_with_status(ls::Linesearch{T,<:Quadratic}, α₀::T, params=NullParameters()) where {T}
    φ₀ = value(problem(ls), zero(T), params)
    d₀anchor = derivative(problem(ls), zero(T), params)

    anchor = check_anchor(φ₀, d₀anchor, α₀)
    isnothing(anchor) || return anchor

    τ = armijo_tolerance(φ₀, T(DEFAULT_ARMIJO_τ_ULPS))
    αres = _quadratic_search(ls, α₀, params)

    # `bracket_minimum_with_fixed_point` flips direction when the merit rises to the right of
    # the bracketing *start* — which is α₀, not 0 — so even a decreasing anchor can yield a
    # bracket, and hence a minimiser, left of zero once α₀ overshoots. A negative step is not a
    # meaningful step length along a direction (see the α > 0 contract), so retry once from the
    # α = 0 anchor, which `check_anchor` has established is decreasing.
    if isnothing(αres) || αres ≤ zero(T)
        αres = _quadratic_search(ls, zero(T), params)
    end
    isnothing(αres) && return LinesearchStatus{T}(α₀, LINESEARCH_FLOOR, 0, φ₀, d₀anchor, φ₀, τ, zero(T))
    αres > zero(T) || return LinesearchStatus{T}(α₀, LINESEARCH_NO_DESCENT, 0, φ₀, d₀anchor, φ₀, τ, zero(T))

    φres = value(problem(ls), αres, params)
    LinesearchStatus{T}(αres, φres ≤ φ₀ - τ ? LINESEARCH_DECREASED : LINESEARCH_FLOOR,
        0, φ₀, d₀anchor, φres, τ, zero(T))
end

# The quadratic-fit iteration itself. Returns `nothing` if the merit cannot be bracketed.
# Private: `solve`/`solve_with_status` is the public entry point.
function _quadratic_search(ls::Linesearch{T,<:Quadratic}, α₀::T, params) where {T}
    # Start the bracketing at the caller's α₀ when it lies on the descent side
    # (φ′(α₀) < 0, so the minimiser is to its right); otherwise keep the α = 0 anchor,
    # where a descent direction is guaranteed decreasing. `bracket_minimum_with_fixed_point`
    # searches rightward from a fixed left point, so that point must be on the descent
    # side. See issue #164.
    α = (α₀ > zero(T) && derivative(problem(ls), α₀, params) < zero(T)) ? α₀ : zero(T)
    s = method(ls).s

    for _ in 1:config(ls).linesearch_max_iterations
        # fit p(α) = p₀ + p₁(α - a) + p₂(α - a)² with p₀ = y₀, p₁ = d₀ and
        # p₂ = (y₁ - y₀ - d₀(b - a)) / (b - a)²; the endpoint merits y₀, y₁ come
        # from the bracketing, so no re-evaluation is needed here.
        bracket = bracket_minimum_with_fixed_point(problem(ls), params, α, s)
        # `nothing` means the merit could not be bracketed from here (see `bracket_minimum`).
        isnothing(bracket) && return nothing
        a, b, y₀, y₁ = bracket
        d₀ = derivative(problem(ls), a, params)
        # `d₀` is the derivative at the bracket's left endpoint `a`; return that point
        # (not the loop's start `α`), which differ when the bracketer flipped because
        # the start was not on the descent side.
        abs(d₀) < method(ls).ε && return a

        # minimizer αₜ = a - p₁ / (2p₂); guard on the fitted curvature (denom = 2p₂(b-a)²).
        # A non-positive curvature (denom ≤ 0), a non-finite αₜ, or a minimizer outside
        # the bracket means the quadratic model is untrustworthy — bisect [a, b] instead.
        denom = 2 * (y₁ - y₀ - d₀ * (b - a))
        αₜ = denom > zero(T) ? a - d₀ * (b - a)^2 / denom : (a + b) / 2
        (isfinite(αₜ) && a ≤ αₜ ≤ b) || (αₜ = (a + b) / 2)

        (l2norm(αₜ - α) < method(ls).ε) && return αₜ

        α = αₜ
        s *= method(ls).s_reduction
    end

    α
end

Base.show(io::IO, ls::Quadratic) = print(io, "Quadratic Polynomial with ε = $(ls.ε), s = $(ls.s) and s_reduction = $(ls.s_reduction).")

function change_precision(::Type{T}, method::Quadratic) where {T}
    T ≠ eltype(method) || return method
    Quadratic{T}(T(method.ε), T(method.s), T(method.s_reduction))
end

function Base.isapprox(qu₁::Quadratic{T}, qu₂::Quadratic{T}; kwargs...) where {T}
    isapprox(qu₁.ε, qu₂.ε; kwargs...) && isapprox(qu₁.s, qu₂.s; kwargs...) && isapprox(qu₁.s_reduction, qu₂.s_reduction; kwargs...)
end
