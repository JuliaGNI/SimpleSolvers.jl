"""
    StaticState <: LinesearchState

The state for [`Static`](@ref).

# Functors

For a `Number` `a` and an [`AbstractUnivariateObjective`](@ref) `obj` we have the following functors:
```julia
ls.(a) = ls.α
ls.(obj, a) = ls.α
```
"""
struct StaticState{T} <: LinesearchState
    α::T
    StaticState(α::T = 1.0) where {T} = new{T}(α)
end

# StaticState(args...; α = 1.0, kwargs...) = StaticState(α)

LinesearchState(algorithm::Static; kwargs...) = StaticState(algorithm.α)

Base.show(io::IO, state::StaticState) = show(io, Static(state.α))

(ls::StaticState)(::Number = 0) = ls.α
(ls::StaticState)(::AbstractUnivariateObjective, ::Number = 0) = ls.α

# (ls::StaticState)(x::AbstractVector, δx::AbstractVector) = x .+= ls.α .* δx