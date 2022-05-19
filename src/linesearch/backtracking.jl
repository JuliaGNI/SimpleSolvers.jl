
using Printf

const DEFAULT_ARMIJO_α₀ = 1.0
const DEFAULT_ARMIJO_σ₀ = 0.1
const DEFAULT_ARMIJO_σ₁ = 0.5
const DEFAULT_ARMIJO_p  = 0.5

struct BacktrackingState{OBJ,OPT,T} <: LinesearchState where {OBJ <: AbstractObjective, OPT <: Options, T <: Number}
    objective::OBJ
    config::OPT

    α₀::T
    ϵ::T
    p::T

    function BacktrackingState(objective; config = Options(),
                    α₀::T = DEFAULT_ARMIJO_α₀,
                    ϵ::T = DEFAULT_WOLFE_ϵ,
                    p::T = DEFAULT_ARMIJO_p) where {T}
        new{typeof(objective), typeof(config), T}(objective, config, α₀, ϵ, p)
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


function (ls::BacktrackingState{<:UnivariateObjective})()
    local α = ls.α₀
    local y₀ = value!(ls.objective, zero(α))
    local d₀ = derivative!(ls.objective, zero(α))

    for _ in 1:ls.config.max_iterations
        if value!(ls.objective, α) ≥ y₀ + ls.ϵ * α * d₀
            α *= ls.p
        else
            break
        end
    end

    return α
end

backtracking(f, x; kwargs...) = BacktrackingState(f, x; kwargs...)()
