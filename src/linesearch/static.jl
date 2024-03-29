
struct StaticState{T} <: LinesearchState
    alpha::T
    StaticState(alpha::T = 1.0) where {T} = new{T}(alpha)
end

StaticState(args...; alpha = 1.0, kwargs...) = StaticState(alpha)

LinesearchState(algorithm::Static; kwargs...) = StaticState(algorithm.alpha)

Base.show(io::IO, ls::StaticState) = print(io, "Static Linesearch")

(ls::StaticState)(x::Number = 0) = ls.alpha
(ls::StaticState)(o::AbstractUnivariateObjective, x::Number = 0) = ls.alpha

# (ls::StaticState)(x::AbstractVector, δx::AbstractVector) = x .+= ls.alpha .* δx
