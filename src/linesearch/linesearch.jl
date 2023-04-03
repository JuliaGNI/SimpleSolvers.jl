
const DEFAULT_LINESEARCH_nmax=100
const DEFAULT_LINESEARCH_rmax=100

const DEFAULT_WOLFE_ϵ  = 1E-4

abstract type LinesearchState end

# LinesearchState(algorithm, f::Callable, x::Number; kwargs...) = LinesearchState(algorithm, UnivariateObjective(f, x); kwargs...)
# LinesearchState(algorithm, f::Callable, g::Callable, x::Number; kwargs...) = LinesearchState(algorithm, UnivariateObjective(f, g, x); kwargs...)
# LinesearchState(algorithm, f::Callable, x::AbstractVector; kwargs...) = LinesearchState(algorithm, MultivariateObjective(f, x); kwargs...)

(ls::LinesearchState)(f::Callable; kwargs...) = ls(TemporaryUnivariateObjective(f, missing); kwargs...)
(ls::LinesearchState)(f::Callable, x::Number; kwargs...) = ls(TemporaryUnivariateObjective(f, missing), x; kwargs...)
(ls::LinesearchState)(f::Callable, g::Callable; kwargs...) = ls(TemporaryUnivariateObjective(f, g); kwargs...)
(ls::LinesearchState)(f::Callable, g::Callable, x::Number; kwargs...) = ls(TemporaryUnivariateObjective(f, g), x; kwargs...)


# solve!(x, δx, ls::LinesearchState) = ls(x, δx)
# solve!(x, δx, g, ls::LinesearchState) = ls(x, δx, g)

struct Linesearch{ALG <: LinesearchMethod, OPT <: Options, OST <: LinesearchState}
    algorithm::ALG
    config::OPT
    state::OST

    function Linesearch(algorithm, config, state)
        new{typeof(algorithm), typeof(config), typeof(state)}(algorithm, config, state)
    end
end

function Linesearch(; algorithm = Static(), config = Options(), kwargs...)
    state = LinesearchState(algorithm; kwargs...)
    Linesearch(algorithm, config, state)
end

# function Linesearch(x::Number, F::Callable; D = nothing, kwargs...)
#     objective = UnivariateObjective(F, D, x)
#     Linesearch(x, objective; kwargs...)
# end

# function Linesearch(x::AbstractVector, F::Callable; D = nothing, kwargs...)
#     objective = MultivariateObjective(F, D, x)
#     Linesearch(x, objective; kwargs...)
# end


(ls::Linesearch)(args...; kwargs...) = ls.state(args...; kwargs...)
# (ls::Linesearch)(f::Callable, args...; kwargs...) = ls(TemporaryUnivariateObjective(f, missing), args...; kwargs...)
# (ls::Linesearch)(f::Callable, g::Callable, args...; kwargs...) = ls(TemporaryUnivariateObjective(f, g), args...; kwargs...)

# solve!(x, δx, ls::Linesearch) = solve!(x, δx, ls.state)
