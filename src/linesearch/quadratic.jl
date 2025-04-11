"""
Quadratic Polynomial line search
"""
struct QuadraticState{OPT,T} <: LinesearchState where {OPT <: Options, T <: Number}
    config::OPT

    α₀::T
    σ₀::T
    σ₁::T
    ϵ::T

    function QuadraticState(; config = Options(),
                    α₀::T = DEFAULT_ARMIJO_α₀,
                    σ₀::T = DEFAULT_ARMIJO_σ₀,
                    σ₁::T = DEFAULT_ARMIJO_σ₁,
                    ϵ::T = DEFAULT_WOLFE_ϵ) where {T}
        new{typeof(config), T}(config, α₀, σ₀, σ₁, ϵ)
    end
end

Base.show(io::IO, ::QuadraticState) = print(io, "Polynomial quadratic")

LinesearchState(algorithm::Quadratic; T::DataType=Float64, kwargs...) = QuadraticState(; kwargs...)

function (ls::QuadraticState)(obj::AbstractUnivariateObjective, α::T = ls.α₀) where {T}
    local αₜ::T
    local y₁::T
    local p₀::T
    local p₁::T
    local p₂::T

    # determine constant coefficients of polynomial p(α) = p₀ + p₁α + p₂α²
    # p₀ = f(x₀)    # value of initial solution
    # p₁ = f'(x₀)   # derivative of initial solution
    y₀ = value!(obj, zero(α))
    d₀ = derivative!(obj, zero(α))
    p₀ = y₀
    p₁ = d₀
    
    for _ in 1:ls.config.max_iterations
        # compute value of new solution
        y₁ = value!(obj, α)

        if y₁ ≥ y₀ + ls.ϵ * α * d₀
            # determine nonconstant coefficient of polynomial p(α) = p₀ + p₁α + p₂α²
            p₂ = (y₁^2 - p₀ - p₁*α) / α^2

            # compute minimum αₜ of p(α)
            αₜ = - p₁ / (2p₂)

            if αₜ < ls.σ₀ * α
                α = ls.σ₀ * α
            elseif αₜ > ls.σ₁ * α
                α = ls.σ₁ * α
            else
                α = αₜ
            end
        else
            break
        end
    end

    return α
end

quadratic(o::AbstractUnivariateObjective, args...; kwargs...) = QuadraticState(; kwargs...)(o, args...)
quadratic(f::Callable, g::Callable, args...; kwargs...) = QuadraticState(; kwargs...)(TemporaryUnivariateObjective(f, g), args...)
