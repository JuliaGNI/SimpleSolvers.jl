
const DEFAULT_LINESEARCH_nmax=100
const DEFAULT_LINESEARCH_rmax=100

abstract type LinesearchState end

LinesearchState(algorithm, f::Callable, x::Number; kwargs...) = LinesearchState(algorithm, UnivariateObjective(f, x); kwargs...)
LinesearchState(algorithm, f::Callable, x::AbstractVector; kwargs...) = LinesearchState(algorithm, MultivariateObjective(f, x); kwargs...)

solve!(x, δx, ls::LinesearchState) = ls(x, δx)
solve!(x, δx, g, ls::LinesearchState) = ls(x, δx, g)

struct Linesearch{ALG <: LinesearchMethod, OBJ <: AbstractObjective, OPT <: Options, OST <: LinesearchState}
    algorithm::ALG
    objective::OBJ
    config::OPT
    state::OST

    function Linesearch(algorithm, objective, config, state)
        new{typeof(algorithm), typeof(objective), typeof(config), typeof(state)}(algorithm, objective, config, state)
    end
end

function Linesearch(x, objective::AbstractObjective; algorithm = Static(), config = Options())
    state = LinesearchState(algorithm, objective)
    Linesearch(algorithm, objective, config, state)
end

function Linesearch(x::Number, F::Callable; D = nothing, kwargs...)
    objective = UnivariateObjective(F, D, x)
    Linesearch(x, objective; kwargs...)
end

function Linesearch(x::AbstractVector, F::Callable; D = nothing, kwargs...)
    objective = MultivariateObjective(F, D, x)
    Linesearch(x, objective; kwargs...)
end


(ls::Linesearch)(args...) = ls.state(args...)

solve!(x, δx, ls::Linesearch) = solve!(x, δx, ls.state)
