
using Printf

const DEFAULT_ARMIJO_α₀ = 1.0
const DEFAULT_ARMIJO_σ₀ = 0.1
const DEFAULT_ARMIJO_σ₁ = 0.5
const DEFAULT_ARMIJO_ϵ  = 1E-4
const DEFAULT_ARMIJO_p  = 0.5

struct BacktrackingState{OBJ,OPT,LSC,T} <: LinesearchState where {OBJ <: AbstractObjective, OPT <: Options, LSC <: LinesearchCache, T <: Number}
    objective::OBJ
    config::OPT
    cache::LSC

    α₀::T
    ϵ::T
    p::T

    function BacktrackingState(objective; config = Options(),
                    α₀::T = DEFAULT_ARMIJO_α₀,
                    ϵ::T = DEFAULT_ARMIJO_ϵ,
                    p::T = DEFAULT_ARMIJO_p) where {T}
        cache = LinesearchCache(objective.x_f)
        new{typeof(objective), typeof(config), typeof(cache), T}(objective, config, cache, α₀, ϵ, p)
    end
end

function BacktrackingState(F::Callable, x::Number = DEFAULT_ARMIJO_α₀; D = nothing, kwargs...)
    objective = UnivariateObjective(F, D, x)
    BacktrackingState(objective; kwargs...)
end

function BacktrackingState(F::Callable, x::AbstractVector; D = nothing, kwargs...)
    objective = MultivariateObjective(F, D, x)
    BacktrackingState(objective; kwargs...)
end

Base.show(io::IO, ls::BacktrackingState) = print(io, "Backtracking")

LinesearchState(algorithm::Backtracking, objective; kwargs...) = BacktrackingState(objective; kwargs...)


function (ls::BacktrackingState{<:UnivariateObjective})(xmin::T, xmax::T) where {T <: Number}
    local α = ls.α₀
    local y = value!(ls.objective, α)

    for lsiter in 1:ls.config.max_iterations
        if value!(ls.objective, α) ≥ (one(T) - ls.ϵ * α) * y
            α *= ls.p
        else
            break
        end
    end

    return α
end

(ls::BacktrackingState)(x::Number) = ls(bracket_minimum(ls.objective, x)...)

function (ls::BacktrackingState{<:MultivariateObjective})(x::T, δ::T) where {T <: AbstractVector}
    update!(ls.cache, x, δ, gradient!(ls.objective, x))

    local α = ls.α₀
    local limit = value!(ls.objective, x) + ls.ϵ * α * ls.cache.g' * δ

    ls.cache.x .= x .+ α .* δ

    for lsiter in 1:ls.config.max_iterations
        if value!(ls.objective, ls.cache.x) > limit
            α *= ls.p
            ls.cache.x .= x .+ α .* δ
        else
            break
        end
    end

    x .+= α .* δ
end

(ls::BacktrackingState)(x, δ, args...; kwargs...) = ls(x, δ)

backtracking(f, x, δx; kwargs...) = BacktrackingState(f, x; kwargs...)(x, δx)
