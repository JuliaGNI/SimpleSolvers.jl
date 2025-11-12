"""
This constant is used for [`QuadraticState`](@ref) and [`BierlaireQuadraticState`](@ref).
"""
const MAX_NUMBER_OF_ITERATIONS_FOR_QUADRATIC_LINESEARCH = 20

"""
A factor by which `s` is reduced in each bracketing iteration (see [`bracket_minimum_with_fixed_point`](@ref)).
"""
const DEFAULT_s_REDUCTION = .5

"""
    QuadraticState2 <: LinesearchState

Quadratic Polynomial line search. This is similar to [`QuadraticState`](@ref), but performs multiple iterations in which all parameters ``p_0``, ``p_1`` and ``p_2`` are changed. This is different from [`QuadraticState`](@ref) (taken from [kelley1995iterative](@cite)), where only ``p_2`` is changed. We further do not check the [`SufficientDecreaseCondition`](@ref) but rather whether the derivative is *small enough*.

This algorithm repeatedly builds new quadratic polynomials until a minimum is found (to sufficient accuracy). The iteration may also stop after we reaches the maximum number of iterations (see [`MAX_NUMBER_OF_ITERATIONS_FOR_QUADRATIC_LINESEARCH`](@ref)).

# Keywords

- `config::`[`Options`](@ref)
- `ε`: A constant that checks the *precision*/*tolerance*.
- `s`: A constant that determines the initial interval for bracketing. By default this is [`DEFAULT_BRACKETING_s`](@ref).
- `s_reduction:` A constant that determines the factor by which `s` is decreased in each new *bracketing iteration*.
"""
struct QuadraticState2{T} <: LinesearchState{T}
    config::Options{T}

    ε::T
    s::T
    s_reduction::T

    function QuadraticState2(T₁::DataType=Float64;
                    ε = eps(T₁),
                    s::T = DEFAULT_BRACKETING_s,
                    s_reduction::T = DEFAULT_s_REDUCTION,
                    options_kwargs...) where {T}
        config₁ = Options(T₁; options_kwargs...)
        new{T₁}(config₁, T₁(ε), T₁(s), T₁(s_reduction))
    end
end

Base.show(io::IO, ::QuadraticState2) = print(io, "Polynomial quadratic (second version)")

LinesearchState(algorithm::Quadratic2; T::DataType=Float64, kwargs...) = QuadraticState2(T; kwargs...)

function (ls::QuadraticState2{T})(obj::AbstractUnivariateProblem{T}, number_of_iterations::Integer = 0, x₀::T=zero(T), s::T=ls.s) where {T}
    number_of_iterations != MAX_NUMBER_OF_ITERATIONS_FOR_QUADRATIC_LINESEARCH || return x₀
    # determine coefficients p₀ and p₁ of polynomial p(α) = p₀ + p₁(α - α₀) + p₂(α - α₀)²
    a, b = bracket_minimum_with_fixed_point(obj, x₀; s = s)
    y₀ = value!(obj, a)
    d₀ = derivative!(obj, a)
    !(abs(d₀) < ls.ε) || return x₀
    
    p₀ = y₀
    p₁ = d₀

    # compute value at `b`
    y₁ = value!(obj, b)

    # determine coefficient p₂ of p(α)
    p₂ = (y₁^2 - p₀ - p₁*(b-a)) / (b-a)^2

    # compute minimum αₜ of p(α); i.e. p'(α) = 0.
    αₜ = -p₁ / (2p₂) + a
    !(l2norm(αₜ - x₀) < ls.ε) || return αₜ

    ls(obj, number_of_iterations + 1, αₜ, s * ls.s_reduction)
end

(ls::QuadraticState2{T})(obj::AbstractUnivariateProblem{T}, x₀::T, s::T=ls.s) where {T} = ls(obj, 0, x₀, s)