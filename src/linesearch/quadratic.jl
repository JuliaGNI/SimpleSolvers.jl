"""
This constant is used for [`Quadratic`](@ref) and [`BierlaireQuadratic`](@ref) in double precision.

In single precision we use [`MAX_NUMBER_OF_ITERATIONS_FOR_QUADRATIC_LINESEARCH_SINGLE_PRECISION`](@ref).
"""
const MAX_NUMBER_OF_ITERATIONS_FOR_QUADRATIC_LINESEARCH = 20

"See [`MAX_NUMBER_OF_ITERATIONS_FOR_QUADRATIC_LINESEARCH`](@ref)."
const MAX_NUMBER_OF_ITERATIONS_FOR_QUADRATIC_LINESEARCH_SINGLE_PRECISION = 5

max_number_of_quadratic_linesearch_iterations(::Type{Float32}) = MAX_NUMBER_OF_ITERATIONS_FOR_QUADRATIC_LINESEARCH_SINGLE_PRECISION
max_number_of_quadratic_linesearch_iterations(::Type{Float64}) = MAX_NUMBER_OF_ITERATIONS_FOR_QUADRATIC_LINESEARCH

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
The iteration may also stop after it reaches the maximum number of iterations (see [`MAX_NUMBER_OF_ITERATIONS_FOR_QUADRATIC_LINESEARCH`](@ref)).

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

function solve(ls::Linesearch{T,<:Quadratic}, α₀::T, params=NullParameters()) where {T}
    # Design note (Phase 5, resolving the former "use α₀" TODO): the caller's α₀ is
    # deliberately *not* used as the bracket start.  `bracket_minimum_with_fixed_point`
    # holds its left endpoint fixed and only expands to the right, so it must start
    # where the merit is guaranteed to be decreasing — α = 0 (φ'(0) < 0 for a
    # descent direction).  Anchoring at α₀ > 0 would fail whenever the optimal step
    # is smaller than α₀, and using α₀ as the initial step *size* over-coarsens the
    # bracket and destabilises stiff problems (the tuned `method(ls).s` is required).
    # The step magnitude is therefore governed by the method's `s`, not by α₀.
    α = zero(T)
    s = method(ls).s

    # Iterate rather than recurse (bugs.md §5): the depth is bounded by the
    # iteration maximum either way, but a loop keeps the stack flat and lets
    # the state updates (α, s) read as what they are.
    for _ in 1:max_number_of_quadratic_linesearch_iterations(T)
        # determine coefficients p₀ and p₁ of polynomial p(α) = p₀ + p₁(α - a) + p₂(α - a)².
        # The bracketing already evaluates the merit at both endpoints, so it
        # returns the values along with the bracket — no re-evaluation here.
        a, b, y₀, y₁ = bracket_minimum_with_fixed_point(problem(ls), params, α, s)
        d₀ = derivative(problem(ls), a, params)
        abs(d₀) < method(ls).ε && return α

        # p₀ = y₀
        # p₁ = d₀

        # determine coefficient p₂ of p(α)
        # p₂ = (y₁ - p₀ - p₁*(b-a)) / (b-a)^2

        # compute minimum αₜ of p(α); i.e. p'(α) = 0.
        # αₜ = a - p₁ / (2p₂)

        # The denominator is 2·p₂·(b - a)², proportional to the fitted curvature p₂.
        # If p₂ ≤ 0 (locally linear or non-convex fit) the quadratic model has no
        # interior minimum; if the resulting αₜ is not finite the fit is degenerate.
        # In either case fall back to a bisection step of the current bracket [a, b]
        # instead of producing an Inf/NaN αₜ.  (Note: p₂ is small but positive near
        # convergence, where the interpolation is still valid, so we guard on the sign
        # and finiteness rather than on a magnitude threshold.)
        denom = 2 * (y₁ - y₀ - d₀ * (b - a))
        αₜ = denom > zero(T) ? a - d₀ * (b - a)^2 / denom : (a + b) / 2
        # The minimum of the merit lies inside the bracket [a, b]; a fitted minimizer
        # outside it (or a non-finite one) means the quadratic model is not to be
        # trusted — bisect the bracket instead.
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
