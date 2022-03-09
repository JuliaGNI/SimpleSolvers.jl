
const DEFAULT_LINESEARCH_nmax=100
const DEFAULT_LINESEARCH_rmax=100

abstract type LinesearchState end

LinesearchState(algorithm, f::Callable, x; kwargs...) = LinesearchState(algorithm, UnivariateObjective(f, x); kwargs...)

solve!(x₀, x₁, ls::LinesearchState) = ls(x₀, x₁)
solve!(x, ls::LinesearchState) = ls(x)

# solve!(x, f, g, x₀, x₁, ls::LinesearchState) = ls(x₀, x₁)
# solve!(x, f, g, ls::LinesearchState) = ls(x)
# solve!(x, f, g, x₀, x₁, ls::LinesearchState) = error("solve!(x, f, g, x₀, x₁, ls::LinesearchState) not implemented for line search ", typeof(ls))


struct Linesearch{ALG <: LinesearchMethod, OBJ <: UnivariateObjective, OPT <: Options, OST <: LinesearchState}
    algorithm::ALG
    objective::OBJ
    config::OPT
    state::OST
end

function Linesearch(x, objective::UnivariateObjective; algorithm = Static(), config = Options())
    state = LinesearchState(algorithm, objective)
    Linesearch{typeof(algorithm), typeof(objective), typeof(config), typeof(state)}(algorithm, objective, config, state)
end

function Linesearch(x::Number, F::Callable; D = nothing, kwargs...)
    objective = UnivariateObjective(F, D, x)
    Linesearch(x, objective; kwargs...)
end


(ls::Linesearch)(xmin, xmax) = ls.state(xmin, xmax)
(ls::Linesearch)(x) = ls.state(x)

# solve!(x, f, g, x₀, x₁, ls::Linesearch) = solve!(x, f, g, x₀, x₁, ls.state)
# solve!(x, f, g, ls::Linesearch) = solve!(x, f, g, ls.state)
solve!(x₀, x₁, ls::Linesearch) = solve!(x₀, x₁, ls.state)
solve!(x, ls::Linesearch) = solve!(x, ls.state)
