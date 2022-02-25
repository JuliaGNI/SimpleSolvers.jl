
struct StaticState{T} <: LinesearchState
    alpha::T
    StaticState(alpha::T = 1.0) where {T} = new{T}(alpha)
end

StaticState(args...; alpha = 1.0, kwargs...) = StaticState(alpha)
StaticState(::UnivariateObjective; alpha = 1.0, kwargs...) = StaticState(alpha)

LinesearchState(algorithm::Static, args...; kwargs...) = StaticState(algorithm.alpha)

Base.show(io::IO, ls::StaticState) = print(io, "Static")


(ls::StaticState)(x) = ls.alpha * x
(ls::StaticState)(x₀, x₁) = ls.alpha * x₁

# solve!(x, f, g, x₀, x₁, ls::NoLinesearchState) = ls(x, x₀, x₁)
# solve!(x, f, g, ls::NoLinesearchState) = ls(x)
solve!(x₀, x₁, ls::StaticState) = ls(x₀, x₁)
solve!(x, ls::StaticState) = ls(x)
