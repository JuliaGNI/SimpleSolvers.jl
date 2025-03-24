const DEFAULT_LINESEARCH_NMAX=100
const DEFAULT_LINESEARCH_RMAX=100

"""
    LinesearchState

Abstract type. 

Examples include [`StaticState`](@ref), [`BacktrackingState`](@ref), [`BisectionState`](@ref) and [`QuadraticState`](@ref).

# Implementation

A `struct` that is subtyped from `LinesearchState` needs to implement the functors:

```julia
ls(x; kwargs...)
ls(obj::AbstractUnivariateObjective, x; kwargs...)
```

Additionaly the following function needs to be extended:

```julia
LinesearchState(algorithm::LinesearchMethod; kwargs...)
```

# Functors

The following functors are auxiliary helper functions:

```julia
ls(f::Callable; kwargs...) = ls(TemporaryUnivariateObjective(f, missing); kwargs...)
ls(f::Callable, x::Number; kwargs...) = ls(TemporaryUnivariateObjective(f, missing), x; kwargs...)
ls(f::Callable, g::Callable; kwargs...) = ls(TemporaryUnivariateObjective(f, g); kwargs...)
ls(f::Callable, g::Callable, x::Number; kwargs...) = ls(TemporaryUnivariateObjective(f, g), x; kwargs...)
```
"""
abstract type LinesearchState end

LinesearchState(algorithm::LinesearchMethod; kwags...) = error("LinesearchState not implemented for algorithm $(typeof(algorithm)).")

(ls::LinesearchState)(f::Callable; kwargs...) = ls(TemporaryUnivariateObjective(f, missing); kwargs...)
(ls::LinesearchState)(f::Callable, x::Number; kwargs...) = ls(TemporaryUnivariateObjective(f, missing), x; kwargs...)
(ls::LinesearchState)(f::Callable, g::Callable; kwargs...) = ls(TemporaryUnivariateObjective(f, g); kwargs...)
(ls::LinesearchState)(f::Callable, g::Callable, x::Number; kwargs...) = ls(TemporaryUnivariateObjective(f, g), x; kwargs...)

# solve!(x, δx, ls::LinesearchState) = ls(x, δx)
# solve!(x, δx, g, ls::LinesearchState) = ls(x, δx, g)

"""
    Linesearch

A `struct` that stores the [`LinesearchMethod`](@ref), the [`LinesearchState`](@ref) and [`Options`](@ref).

# Keys

- `algorithm::`[`LinesearchMethod`](@ref)
- `config::`[`Options`](@ref)
- `state::`[`LinesearchState`](@ref)

# Constructors

The following constructors can be used:

```julia
Linesearch(alg, config, state)
Linesearch(; algorithm, config, kwargs...)
```
"""
struct Linesearch{ALG <: LinesearchMethod, OPT <: Options, OST <: LinesearchState}
    algorithm::ALG
    config::OPT
    state::OST
end

function Linesearch(; algorithm = Static(), config = Options(), kwargs...)
    state = LinesearchState(algorithm; kwargs...)
    Linesearch(algorithm, config, state)
end

(ls::Linesearch)(args...; kwargs...) = ls.state(args...; kwargs...)

# solve!(x, δx, ls::Linesearch) = solve!(x, δx, ls.state)
