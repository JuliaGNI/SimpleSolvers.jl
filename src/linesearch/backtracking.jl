
using Printf

const DEFAULT_ARMIJO_α₀ = 1.0
const DEFAULT_ARMIJO_σ₀ = 0.1
const DEFAULT_ARMIJO_σ₁ = 0.5
const DEFAULT_ARMIJO_p  = 0.5

struct BacktrackingState{OPT,T} <: LinesearchState where {OPT <: Options, T <: Number}
    config::OPT

    α₀::T
    ϵ::T
    p::T

    function BacktrackingState(; config = Options(),
                    α₀::T = DEFAULT_ARMIJO_α₀,
                    ϵ::T = DEFAULT_WOLFE_ϵ,
                    p::T = DEFAULT_ARMIJO_p) where {T}
        new{typeof(config), T}(config, α₀, ϵ, p)
    end
end

Base.show(io::IO, ls::BacktrackingState) = print(io, "Backtracking")

LinesearchState(algorithm::Backtracking; kwargs...) = BacktrackingState(; kwargs...)


function (ls::BacktrackingState)(obj::AbstractUnivariateObjective, α = ls.α₀)
    local y₀ = value!(obj, zero(α))
    local d₀ = derivative!(obj, zero(α))

    for _ in 1:ls.config.max_iterations
        if value!(obj, α) ≥ y₀ + ls.ϵ * α * d₀
            α *= ls.p
        else
            break
        end
    end

    return α
end

backtracking(o::AbstractUnivariateObjective, args...; kwargs...) = BacktrackingState(; kwargs...)(o, args...)
backtracking(f::Callable, g::Callable, args...; kwargs...) = BacktrackingState(; kwargs...)(TemporaryUnivariateObjective(f, g), args...)
