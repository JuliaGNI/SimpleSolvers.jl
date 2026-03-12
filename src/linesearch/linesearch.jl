"""
    LinesearchMethod

Examples include [`Static`](@ref), [`Backtracking`](@ref), [`Bisection`](@ref) and [`Quadratic`](@ref).
See these examples for specific information on linesearch algorithms.

# Extended help

A `LinesearchMethod` always has to be used together in [`Linesearch`](@ref) (or with [`solve`](@ref)).
"""
abstract type LinesearchMethod{T} <: NonlinearMethod end

Base.eltype(::LinesearchMethod{T}) where {T} = T
Base.convert(::Type{T}, method::LinesearchMethod{T}) where {T} = method

function Base.convert(::Type, method::LinesearchMethod)
    error("Convert not implemented for $(typeof(method)).")
end


"""
    Linesearch

A `struct` that stores the [`LinesearchMethod`](@ref) and [`Options`](@ref).

# Keys

- `problem::`[`LinesearchProblem`](@ref)
- `method::`[`LinesearchMethod`](@ref)
- `config::`[`Options`](@ref)

# Constructors

The following constructors can be used:

```julia
Linesearch{T}(problem, method, config)
Linesearch(problem, method=Static(); kwargs...)
```
"""
struct Linesearch{T,MET<:LinesearchMethod{T},PT<:LinesearchProblem{T},OPT<:Options{T}}
    problem::PT
    method::MET
    config::OPT
    Linesearch{T}(problem, method, config) where {T} = new{T,typeof(method),typeof(problem),typeof(config)}(problem, method, config)
end

Linesearch(problem::LinesearchProblem{T}, method::LinesearchMethod=Static(); options_kwargs...) where {T} = Linesearch{T}(problem, convert(T, method), Options(T; options_kwargs...))

problem(s::Linesearch) = s.problem
config(s::Linesearch) = s.config
method(s::Linesearch) = s.method


"""
    solve(linesearch, α, params=NullParameters())
    solve(problem, method, α, params=NullParameters())

Minimize the [`LinesearchProblem`](@ref) with the [`LinesearchMethod`](@ref) `method`.

The argument `params` needs to be of an appropriate form expected by the respective [`LinesearchProblem`](@ref).
"""
function solve(::Linesearch{T,MET}, α::T, params=NullParameters()) where {T,MET<:LinesearchMethod{T}}
    error("Solve method missing for $(MET).")
end

function solve(prob::LinesearchProblem, method::LinesearchMethod, α, params=NullParameters(), config::Options=Options())
    solve(Linesearch(prob, method, config), α, params)
end
