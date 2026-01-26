const DEFAULT_LINESEARCH_NMAX=100
const DEFAULT_LINESEARCH_RMAX=100

"""
    LinesearchState

Abstract type.

Examples include [`StaticState`](@ref), [`BacktrackingState`](@ref), [`BisectionState`](@ref) and [`QuadraticState`](@ref).

# Constructors

The following is used to construct a specific line search state based on a [`LinesearchMethod`](@ref):

```julia
LinesearchState(algorithm::LinesearchMethod; T::DataType=Float64, kwargs...)
```
where the data type should be specified each time the constructor is called. This is done automatically when calling the constructor of [`NewtonSolver`](@ref) for example.
"""
abstract type LinesearchState{T} end

LinesearchState(algorithm::LinesearchMethod; kwags...) = error("LinesearchState not implemented for algorithm $(typeof(algorithm)).")

(ls::LinesearchState)(f::Callable; kwargs...) = ls(LinesearchProblem(f, missing); kwargs...)
(ls::LinesearchState)(f::Callable, x::Number; kwargs...) = ls(LinesearchProblem(f, missing), x; kwargs...)
(ls::LinesearchState)(f::Callable, g::Callable; kwargs...) = ls(LinesearchProblem(f, g); kwargs...)
(ls::LinesearchState)(f::Callable, g::Callable, x::Number; kwargs...) = ls(LinesearchProblem(f, g), x; kwargs...)

"""
    Linesearch

A `struct` that stores the [`LinesearchMethod`](@ref) and [`Options`](@ref).

# Keys

- `algorithm::`[`LinesearchMethod`](@ref)
- `config::`[`Options`](@ref)

# Constructors

The following constructors can be used:

```julia
Linesearch(alg, config)
Linesearch(; algorithm, config, kwargs...)
```
"""
struct Linesearch{ALG <: LinesearchMethod, OPT <: Options}
    algorithm::ALG
    config::OPT
end

function Linesearch(T::DataType=Float64; algorithm = Static(), options_kwargs...)
    config = Options(T; options_kwargs...)
    Linesearch(algorithm, config)
end