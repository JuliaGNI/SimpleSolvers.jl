"""
Quadratic Polynomial line search
"""
struct QuadraticState{OBJ,OPT,T} <: LinesearchState where {OBJ <: AbstractObjective, OPT <: Options, T <: Number}
    objective::OBJ
    config::OPT

    α₀::T
    σ₀::T
    σ₁::T
    ϵ::T

    function QuadraticState(objective; config = Options(),
                    α₀::T = DEFAULT_ARMIJO_α₀,
                    σ₀::T = DEFAULT_ARMIJO_σ₀,
                    σ₁::T = DEFAULT_ARMIJO_σ₁,
                    ϵ::T = DEFAULT_WOLFE_ϵ) where {T}
        new{typeof(objective), typeof(config), T}(objective, config, α₀, σ₀, σ₁, ϵ)
    end
end

# function QuadraticState(F::Callable, x::AbstractVector; D = nothing, kwargs...)
#     objective = MultivariateObjective(F, D, x)
#     QuadraticState(objective; kwargs...)
# end

function QuadraticState(F::Callable, x::Number; D = nothing, kwargs...)
    objective = UnivariateObjective(F, D, x)
    QuadraticState(objective; kwargs...)
end

Base.show(io::IO, ls::QuadraticState) = print(io, "Polynomial quadratic")

LinesearchState(algorithm::Quadratic, objective; kwargs...) = QuadraticState(objective; kwargs...)

function (ls::QuadraticState{<:UnivariateObjective})()
    T = typeof(ls.α₀)
    local α::T = ls.α₀
    local αₜ::T
    local y₁::T
    local p₀::T
    local p₁::T
    local p₂::T

    # determine constant coefficients of polynomial p(α) = p₀ + p₁α + p₂α²
    # p₀ = f(x₀)    # value of initial solution
    # p₁ = f'(x₀)   # derivative of initial solution
    y₀ = value!(ls.objective, zero(α))
    d₀ = derivative!(ls.objective, zero(α))
    p₀ = y₀
    p₁ = d₀
    
    for _ in 1:ls.config.max_iterations
        # compute value of new solution
        y₁ = value!(ls.objective, α)

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

quadratic(f, x; kwargs...) = QuadraticState(f, x; kwargs...)()
