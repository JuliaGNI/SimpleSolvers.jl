const DEFAULT_LINESEARCH_NMAX=100
const DEFAULT_LINESEARCH_RMAX=100

"""
    LinesearchState

Abstract type. 

Examples include [`StaticState`](@ref), [`BacktrackingState`](@ref), [`BisectionState`](@ref) and [`QuadraticState2`](@ref).

# Implementation

A `struct` that is subtyped from `LinesearchState` needs to implement the functors:

```julia
ls(x; kwargs...)
ls(obj::LinesearchProblem, x; kwargs...)
```

Additionaly the following function needs to be extended:

```julia
LinesearchState(algorithm::LinesearchMethod; kwargs...)
```

# Constructors

The following is used to construct a specific line search state based on a [`LinesearchMethod`](@ref):

```julia
LinesearchState(algorithm::LinesearchMethod; T::DataType=Float64, kwargs...)
```
where the data type should be specified each time the constructor is called. This is done automatically when calling the constructor of [`NewtonSolver`](@ref) for example.

# Functors

The following functors are auxiliary helper functions:

```julia
ls(f::Callable; kwargs...) = ls(LinesearchProblem(f, missing); kwargs...)
ls(f::Callable, x::Number; kwargs...) = ls(LinesearchProblem(f, missing), x; kwargs...)
ls(f::Callable, g::Callable; kwargs...) = ls(LinesearchProblem(f, g); kwargs...)
ls(f::Callable, g::Callable, x::Number; kwargs...) = ls(LinesearchProblem(f, g), x; kwargs...)
```
"""
abstract type LinesearchState{T} end

LinesearchState(algorithm::LinesearchMethod; kwags...) = error("LinesearchState not implemented for algorithm $(typeof(algorithm)).")

(ls::LinesearchState)(f::Callable; kwargs...) = ls(LinesearchProblem(f, missing); kwargs...)
(ls::LinesearchState)(f::Callable, x::Number; kwargs...) = ls(LinesearchProblem(f, missing), x; kwargs...)
(ls::LinesearchState)(f::Callable, g::Callable; kwargs...) = ls(LinesearchProblem(f, g); kwargs...)
(ls::LinesearchState)(f::Callable, g::Callable, x::Number; kwargs...) = ls(LinesearchProblem(f, g), x; kwargs...)

# TODO: clarify why we need the extra struct `LineSearch`. Are `LinesearchMethod`s together with `LinesearchState` not enough?
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

function Linesearch(T::DataType=Float64; algorithm = Static(), options_kwargs...)
    state = LinesearchState(algorithm; options_kwargs...)
    config = Options(T; options_kwargs...)
    Linesearch(algorithm, config, state)
end

(ls::Linesearch)(args...; kwargs...) = ls.state(args...; kwargs...)