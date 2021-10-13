
abstract type LineSearch end

struct NoLineSearch <: LineSearch end


solve!(x, x₀, x₁, ls::LineSearch) = error("solve!(x, x₀, x₁, ls::LineSearch) not implemented for line search ", typeof(ls))

