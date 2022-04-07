"""
Quadratic Armijo line search
"""
struct ArmijoQuadraticState{OBJ,OPT,LSC,T} <: LinesearchState where {OBJ <: MultivariateObjective, OPT <: Options, LSC <: LinesearchCache, T <: Number}
    objective::OBJ
    config::OPT
    cache::LSC

    α₀::T
    σ₀::T
    σ₁::T
    ϵ::T

    function ArmijoQuadraticState(objective::MultivariateObjective; config = Options(),
                    α₀::T=DEFAULT_ARMIJO_α₀, σ₀::T=DEFAULT_ARMIJO_σ₀, σ₁::T=DEFAULT_ARMIJO_σ₁, ϵ::T=DEFAULT_ARMIJO_ϵ) where {T}
        cache = LinesearchCache(objective.x_f)
        new{typeof(objective), typeof(config), typeof(cache), T}(objective, config, cache, α₀, σ₀, σ₁, ϵ)
    end
end

function ArmijoQuadraticState(F::Callable, x::AbstractVector; D = nothing, kwargs...)
    objective = MultivariateObjective(F, D, x)
    ArmijoQuadraticState(objective; kwargs...)
end

Base.show(io::IO, ls::ArmijoQuadraticState) = print(io, "Armijo (quadratic)")

LinesearchState(algorithm::ArmijoQuadratic, objective; kwargs...) = ArmijoQuadraticState(objective; kwargs...)


function (ls::ArmijoQuadraticState)(x::T, δ::T) where {T <: AbstractVector}
    update!(ls.cache, x, δ, gradient!(ls.objective, x))

    local α::T = ls.α₀
    local αₜ::T
    local y₀::T
    local y₁::T
    local p₀::T
    local p₁::T
    local p₂::T


    # compute norms of initial solution
    y₀ = value!(ls.objective, x)

    # δy = Jδx
    mul!(ls.δy, ls.cache.g, ls.cache.δx) # TODO !!!

    # determine constant coefficients of polynomial p(α) = p₀ + p₁α + p₂α²
    p₀ = y₀^2
    # p₁ = 2(f ⋅ ls.δy) # TODO !!!
    
    for lsiter in 1:ls.config.max_iterations
        # compute norms of solution
        y₁ = value!(ls.objective, α)

        if y₁ ≥ (one(T) - ls.ϵ * α) * y₀
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

(ls::ArmijoQuadraticState)(x, δ, args...; kwargs...) = ls(x, δ)

armijo_quadratic(f, x, δx; kwargs...) = ArmijoQuadraticState(f, x; kwargs...)(x, δx)
