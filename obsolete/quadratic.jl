@doc raw"""
    adjust_α(ls, αₜ, α)

Check which conditions the new `αₜ` is in ``[\sigma_0\alpha_0, \simga_1\alpha_0]`` and return the updated `α` accordingly (it is updated if it does not lie in the interval).

We first check the following:
```math
    \alpha_t  < \alpha_0\alpha,
```
where ``\sigma_0`` is stored in `ls` (i.e. in an instance of [`QuadraticState`](@ref)).
If this is not true we check:
```math
    \alpha_t > \sigma_1\alpha,
```
where ``\sigma_1`` is again stored in `ls`. If this second condition is also not true we simply return the unchanged ``\alpha_t``.
So if `\alpha_t` does not lie in the interval ``(\sigma_0\alpha, \sigma_1\alpha)`` the interval is made bigger by either multiplying with ``\sigma_0`` (default [`DEFAULT_ARMIJO_σ₀`](@ref)) or ``\sigma_1`` (default [`DEFAULT_ARMIJO_σ₁`](@ref)).

!!! warning
    This was used for the old `Quadratic` line search and seems to be not used anymore for `Quadratic` and other line searches.
"""
function adjust_α(ls::QuadraticState{T}, αₜ::T, α::T) where {T}
    adjust_α(ls.σ₀, ls.σ₁, αₜ, α)
end

"""
    QuadraticState <: LinesearchState

Quadratic Polynomial line search.

*Quadratic line search* works by fitting a polynomial to a univariate problem (see [`LinesearchProblem`](@ref)) and then finding the minimum of that polynomial. Also compare this to [`BierlaireQuadraticState`](@ref). The algorithm is taken from [kelley1995iterative](@cite).

# Keywords

- `config::`[`Options`](@ref)
- `α₀`: by default [`DEFAULT_ARMIJO_α₀`](@ref)
- `σ₀`: by default [`DEFAULT_ARMIJO_σ₀`](@ref)
- `σ₁`: by default [`DEFAULT_ARMIJO_σ₁`](@ref)
- `c`: by default [`DEFAULT_WOLFE_c₁`](@ref)
"""
struct QuadraticState{T} <: LinesearchState{T}
    config::Options{T}

    α₀::T
    σ₀::T
    σ₁::T
    c::T

    function QuadraticState(T₁::DataType=Float64;
                    α₀::T = DEFAULT_ARMIJO_α₀,
                    σ₀::T = DEFAULT_ARMIJO_σ₀,
                    σ₁::T = DEFAULT_ARMIJO_σ₁,
                    c::T = DEFAULT_WOLFE_c₁,
                    options_kwargs...) where {T}
        config₁ = Options(T₁; options_kwargs...)
        new{T₁}(config₁, T₁(α₀), T₁(σ₀), T₁(σ₁), T₁(c))
    end
end

Base.show(io::IO, ::QuadraticState) = print(io, "Polynomial quadratic")

LinesearchState(algorithm::Quadratic; T::DataType=Float64, kwargs...) = QuadraticState(T; kwargs...)

function (ls::QuadraticState{T})(obj::LinesearchProblem{T}, number_of_iterations::Integer = 0, α::T = ls.α₀, x₀::T=zero(T)) where {T}
    number_of_iterations != ls.config.max_iterations || error("Maximum number of iterations reached when doing quadratic line search.")
    # determine constant coefficients of polynomial p(α) = p₀ + p₁α + p₂α²
    y₀ = value(obj, x₀)
    d₀ = derivative!(obj, x₀)
    p₀ = y₀
    p₁ = d₀
    α = determine_initial_α(obj, α, y₀)

    sdc = SufficientDecreaseCondition(ls.c, x₀, y₀, d₀, one(T), obj)
    # compute value of new solution
    y₁ = value(obj, α)

    # determine nonconstant coefficient of polynomial p(α) = p₀ + p₁α + p₂α²
    p₂ = (y₁^2 - p₀ - p₁*(α-x₀)) / (α-x₀)^2

    # compute minimum αₜ of p(α); i.e. p'(α) = 0.
    αₜ = - p₁ / (2p₂) + x₀

    α = adjust_α(ls, αₜ, α)

    sdc(α) ? α : ls(obj, number_of_iterations + 1, α * DEFAULT_ARMIJO_p, x₀)
end
