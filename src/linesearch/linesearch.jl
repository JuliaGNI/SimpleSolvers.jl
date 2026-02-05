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

- `method::`[`LinesearchMethod`](@ref)
- `config::`[`Options`](@ref)

# Constructors

The following constructors can be used:

```julia
Linesearch{T}(method, config)
Linesearch(T; method, kwargs...)
Linesearch(method; T, kwargs...)
Linesearch(; T, kwargs...)
```
"""
struct Linesearch{T,MET<:LinesearchMethod{T},OPT<:Options{T}}
    method::MET
    config::OPT
    Linesearch{T}(method, config) where {T} = new{T,typeof(method),typeof(config)}(method, config)
end

Linesearch(T::DataType, method::LinesearchMethod=Static(), options_kwargs...) = Linesearch{T}(convert(T, method), Options(T; options_kwargs...))
Linesearch(method::LinesearchMethod{T}; options_kwargs...) where {T} = Linesearch{T}(method, Options(T; options_kwargs...))

config(s::Linesearch) = s.config
method(s::Linesearch) = s.method


"""
    solve(problem, linesearch, α, params=NullParameters())
    solve(problem, method, α, params=NullParameters())

Minimize the [`LinesearchProblem`](@ref) with the [`LinesearchMethod`](@ref) `method`.
"""
function solve(::LinesearchProblem{T}, ::Linesearch{T,MET}, α::T, params=NullParameters()) where {T,MET<:LinesearchMethod{T}}
    error("Solve method missing for $(MET).")
end

function solve(prob::LinesearchProblem, method::LinesearchMethod, α, params=NullParameters(), config::Options=Options())
    solve(prob, Linesearch(method, config), α, params=NullParameters())
end
