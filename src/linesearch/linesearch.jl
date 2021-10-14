
abstract type LineSearch end

solve!(x, δx, x₀, y₀, g₀, ls::LineSearch) = error("solve!(x, δx, x₀, y₀, g₀, ls::LineSearch) not implemented for line search ", typeof(ls))
