"""
    determine_initial_α(y₀, obj, α₀)

Check whether `α₀` satisfies the [`BracketMinimumCriterion`](@ref) for `obj`. If the criterion is not satisfied we call [`bracket_minimum_with_fixed_point`](@ref).
This is used as a starting point for using the functor of [`Quadratic`](@ref) and makes sure that `α` describes *a point past the minimum*.

!!! warning
    This was used for the old `Quadratic` line search and seems to be not used anymore for `Quadratic` and other line searches.
"""
function determine_initial_α(obj::LinesearchProblem, α₀::T, x₀::T=zero(T), y₀::T=value(obj, x₀)) where {T}
    if derivative(obj, x₀) < zero(T)
        BracketMinimumCriterion()(y₀, value(obj, x₀ + α₀)) ? α₀ : bracket_minimum_with_fixed_point(obj, x₀)[2]
    else
        bracket_minimum_with_fixed_point(obj, x₀)[1]
    end
end

"""
This constant is used for [`Quadratic`](@ref) and [`BierlaireQuadratic`](@ref).
"""
const MAX_NUMBER_OF_ITERATIONS_FOR_QUADRATIC_LINESEARCH = 20

const MAX_NUMBER_OF_ITERATIONS_FOR_QUADRATIC_LINESEARCH_SINGLE_PRECISION = 5

max_number_of_quadratic_linesearch_iterations(::Type{Float32}) = MAX_NUMBER_OF_ITERATIONS_FOR_QUADRATIC_LINESEARCH_SINGLE_PRECISION
max_number_of_quadratic_linesearch_iterations(::Type{Float64}) = MAX_NUMBER_OF_ITERATIONS_FOR_QUADRATIC_LINESEARCH

"""
A factor by which `s` is reduced in each bracketing iteration (see [`bracket_minimum_with_fixed_point`](@ref)).
"""
const DEFAULT_s_REDUCTION = .5

"""
    Quadratic <: LinesearchMethod

Quadratic Polynomial line search. Performs multiple iterations in which all parameters ``p_0``, ``p_1`` and ``p_2`` are changed. This is different from the old `Quadratic` (taken from [kelley1995iterative](@cite)), where only ``p_2`` is changed. We further do not check the [`SufficientDecreaseCondition`](@ref) but rather whether the derivative is *small enough*.

!!! warning
    The old `Quadratic` was deprecated!

This algorithm repeatedly builds new quadratic polynomials until a minimum is found (to sufficient accuracy). The iteration may also stop after we reaches the maximum number of iterations (see [`MAX_NUMBER_OF_ITERATIONS_FOR_QUADRATIC_LINESEARCH`](@ref)).

# Keywords

- `config::`[`Options`](@ref)
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

    function Quadratic(T₁::DataType=Float64;
                    ε = eps(T₁),
                    s::T = DEFAULT_BRACKETING_s,
                    s_reduction::T = DEFAULT_s_REDUCTION) where {T}
        new{T₁}(T₁(ε), T₁(s), T₁(s_reduction))
    end
end

function solve(problem::LinesearchProblem{T}, ls::Linesearch{T, LST}, number_of_iterations::Integer = 0, x₀::T=zero(T), s::T=ls.algorithm.s) where {T, LST <: Quadratic}
    number_of_iterations != max_number_of_quadratic_linesearch_iterations(T) || return x₀
    # determine coefficients p₀ and p₁ of polynomial p(α) = p₀ + p₁(α - α₀) + p₂(α - α₀)²
    a, b = bracket_minimum_with_fixed_point(problem, x₀; s = s)
    y₀ = value(problem, a)
    d₀ = derivative(problem, a)
    !(abs(d₀) < ls.algorithm.ε) || return x₀

    p₀ = y₀
    p₁ = d₀

    # compute value at `b`
    y₁ = value(problem, b)

    # determine coefficient p₂ of p(α)
    p₂ = (y₁^2 - p₀ - p₁*(b-a)) / (b-a)^2

    # compute minimum αₜ of p(α); i.e. p'(α) = 0.
    αₜ = -p₁ / (2p₂) + a
    !(l2norm(αₜ - x₀) < ls.algorithm.ε) || return αₜ

    solve(problem, ls, number_of_iterations + 1, αₜ, s * ls.algorithm.s_reduction)
end

solve(problem::LinesearchProblem{T}, ls::Linesearch{T, LST}, x₀::T, s::T=ls.s) where {T, LST <: Quadratic} = solve(problem, ls, 0, x₀, s)

Base.show(io::IO, ::Quadratic) = print(io, "Quadratic Polynomial")

function Base.convert(::Type{T}, algorithm::Quadratic) where {T}
    T ≠ eltype(algorithm) || return algorithm
    Quadratic(T; ε=T(algorithm.ε), s=T(algorithm.s), s_reduction=T(algorithm.s_reduction))
end