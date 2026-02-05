"""
    determine_initial_α(y₀, obj, α₀)

Check whether `α₀` satisfies the [`BracketMinimumCriterion`](@ref) for `obj`. If the criterion is not satisfied we call [`bracket_minimum_with_fixed_point`](@ref).
This is used as a starting point for using the functor of [`Quadratic`](@ref) and makes sure that `α` describes *a point past the minimum*.

!!! warning
    This was used for the old `Quadratic` line search and seems to be not used anymore for `Quadratic` and other line searches.
"""
function determine_initial_α(problem::LinesearchProblem, α₀::T, x₀::T=zero(T), y₀::T=value(problem, x₀)) where {T}
    if derivative(problem, x₀) < zero(T)
        BracketMinimumCriterion()(y₀, value(problem, x₀ + α₀)) ? α₀ : bracket_minimum_with_fixed_point(problem.F, problem.D, x₀)[2]
    else
        bracket_minimum_with_fixed_point(problem.F, problem.D, x₀)[1]
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
const DEFAULT_s_REDUCTION = 0.5

"""
    Quadratic <: LinesearchMethod

Quadratic Polynomial line search based on the polynomial p(α) = p₀ + p₁(α - α₀) + p₂(α - α₀)².
Performs multiple iterations in which all parameters ``p_0``, ``p_1`` and ``p_2`` are adapted.
We do not check the [`SufficientDecreaseCondition`](@ref) but rather whether the derivative is *sufficiently small*.

This algorithm repeatedly builds new quadratic polynomials until a minimum is found (to sufficient accuracy).
The iteration may also stop after we reaches the maximum number of iterations (see [`MAX_NUMBER_OF_ITERATIONS_FOR_QUADRATIC_LINESEARCH`](@ref)).

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

    function Quadratic{T}(ε::T, s::T, s_reduction::T) where {T}
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

Quadratic(::Type{T}, ::NonlinearSolverMethod) where {T} = Quadratic{T}(
    default_precision(T)^2,
    T(DEFAULT_BRACKETING_s^2),
    T(DEFAULT_s_REDUCTION^2)
)

Quadratic(::Type{T}, ::OptimizerMethod) where {T} = Quadratic(T)


function solve(problem::LinesearchProblem{T}, ls::Linesearch{T,<:Quadratic}, α₀::T, params, s::T, number_of_iterations::Integer) where {T}
    number_of_iterations ≤ max_number_of_quadratic_linesearch_iterations(T) || return α₀

    # determine coefficients p₀ and p₁ of polynomial p(α) = p₀ + p₁(α - α₀) + p₂(α - α₀)²
    a, b = bracket_minimum_with_fixed_point(problem.F, problem.D, α₀, s)
    d₀ = derivative(problem, a)
    !(abs(d₀) < method(ls).ε) || return α₀

    # compute values at `a` and `b`
    y₀ = value(problem, a)
    y₁ = value(problem, b)

    # p₀ = y₀
    # p₁ = d₀

    # determine coefficient p₂ of p(α)
    # p₂ = (y₁ - p₀ - p₁*(b-a)) / (b-a)^2

    # compute minimum αₜ of p(α); i.e. p'(α) = 0.
    # αₜ = a - p₁ / (2p₂)

    αₜ = a - d₀ * (b - a)^2 / 2(y₁ - y₀ - d₀ * (b - a))

    !(l2norm(αₜ - α₀) < method(ls).ε) || return αₜ

    solve(problem, ls, αₜ, params, s * method(ls).s_reduction, number_of_iterations + 1)
end

function solve(problem::LinesearchProblem{T}, ls::Linesearch{T,<:Quadratic}, α₀::T, params=NullParameters()) where {T}
    # TODO: The following line should use α₀ instead of zero(T) but that requires a rework of the bracketing algorithm
    # solve(problem, ls, α₀, params, method(ls).s, 0)
    solve(problem, ls, zero(T), params, method(ls).s, 0)
end

Base.show(io::IO, ls::Quadratic) = print(io, "Quadratic Polynomial with ε = $(ls.ε), s = $(ls.s) and s_reduction = $(ls.s_reduction).")

function Base.convert(::Type{T}, method::Quadratic) where {T}
    T ≠ eltype(method) || return method
    Quadratic{T}(T(method.ε), T(method.s), T(method.s_reduction))
end

function Base.isapprox(qu₁::Quadratic{T}, qu₂::Quadratic{T}; kwargs...) where {T}
    isapprox(qu₁.ε, qu₂.ε; kwargs...) && isapprox(qu₁.s, qu₂.s; kwargs...) && isapprox(qu₁.s_reduction, qu₂.s_reduction; kwargs...)
end
