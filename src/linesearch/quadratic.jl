"""
    QuadraticState <: LinesearchState

Quadratic Polynomial line search.

*Quadratic line search* works by fitting a polynomial to a univariate objective (see [`AbstractUnivariateObjective`](@ref)) and then finding the minimum of that polynomial.

# Keywords

- `config::`[`Options`](@ref)
- `α₀`: by default [`DEFAULT_ARMIJO_α₀`](@ref)
- `σ₀`: by default [`DEFAULT_ARMIJO_σ₀`](@ref)
- `σ₁`: by default [`DEFAULT_ARMIJO_σ₁`](@ref)
- `c`: by default [`DEFAULT_WOLFE_c₁`](@ref)
"""
struct QuadraticState{T,OPT} <: LinesearchState where {T <: Number, OPT <: Options{T}}
    config::OPT

    α₀::T
    σ₀::T
    σ₁::T
    c::T

    function QuadraticState(T₁::DataType=Float64; config = Options(),
                    α₀::T = DEFAULT_ARMIJO_α₀,
                    σ₀::T = DEFAULT_ARMIJO_σ₀,
                    σ₁::T = DEFAULT_ARMIJO_σ₁,
                    c::T = DEFAULT_WOLFE_c₁) where {T}
        config₁ = Options(T₁, config)
        new{T₁, typeof(config₁)}(config₁, T₁(α₀), T₁(σ₀), T₁(σ₁), T₁(c))
    end
end

Base.show(io::IO, ::QuadraticState) = print(io, "Polynomial quadratic")

LinesearchState(algorithm::Quadratic; T::DataType=Float64, kwargs...) = QuadraticState(T; kwargs...)

@doc raw"""
    adjust_alpha(ls, αₜ, α)

Check which conditions the new `αₜ` satisfies and return the updated `α` accordingly.

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
"""
function adjust_alpha(ls::QuadraticState{T}, αₜ::T, α::T) where {T}
    if αₜ < ls.σ₀ * α
        ls.σ₀ * α
    elseif αₜ > ls.σ₁ * α
        ls.σ₁ * α
    else
        αₜ
    end
end

function (ls::QuadraticState{T})(obj::AbstractUnivariateObjective{T}, α::T = ls.α₀) where {T}
    # determine constant coefficients of polynomial p(α) = p₀ + p₁α + p₂α²
    y₀ = value!(obj, zero(T))
    d₀ = derivative!(obj, zero(T))
    p₀ = y₀
    p₁ = d₀
    
    sdc = SufficientDecreaseCondition(ls.c, zero(T), y₀, d₀, one(T), obj)
    for _ in 1:ls.config.max_iterations
        # compute value of new solution
        y₁ = value!(obj, α)

        if sdc(α)
            break
        else
            # determine nonconstant coefficient of polynomial p(α) = p₀ + p₁α + p₂α²
            p₂ = (y₁^2 - p₀ - p₁*α) / α^2

            # compute minimum αₜ of p(α); i.e. p'(α) = 0.
            αₜ = - p₁ / (2p₂)

            α = adjust_alpha(ls, αₜ, α)
        end
    end

    α
end

quadratic(o::AbstractUnivariateObjective, args...; kwargs...) = QuadraticState(; kwargs...)(o, args...)
quadratic(f::Callable, g::Callable, args...; kwargs...) = QuadraticState(; kwargs...)(TemporaryUnivariateObjective(f, g), args...)
