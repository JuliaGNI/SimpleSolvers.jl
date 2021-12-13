
struct NoLineSearch <: LineSearch end

NoLineSearch(::Any, ::AbstractArray, ::AbstractArray) = NoLineSearch()

(ls::NoLineSearch)(x, x₀, x₁) where {T} = x .= x₁

(ls::NoLineSearch)(x, f, g, x₀, x₁) = ls(x, x₀, x₁)

solve!(x, f, g, x₀, x₁, ls::NoLineSearch) = ls(x, x₀, x₁)
solve!(x, x₀, x₁, ls::NoLineSearch) = ls(x, x₀, x₁)
