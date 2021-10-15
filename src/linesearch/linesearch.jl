
const DEFAULT_LINESEARCH_nmax=100
const DEFAULT_LINESEARCH_rmax=100

abstract type LineSearch end

solve!(x, f, g, x₀, x₁, ls::LineSearch) = error("solve!(x, f, g, x₀, x₁, ls::LineSearch) not implemented for line search ", typeof(ls))
