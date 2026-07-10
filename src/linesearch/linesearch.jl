"""
    LinesearchMethod{T} <: SolverMethod

Examples include [`Static`](@ref), [`Backtracking`](@ref), [`Bisection`](@ref) , [`BierlaireQuadratic`](@ref) and [`Quadratic`](@ref).
See these examples for specific information on linesearch algorithms.

# Extended help

A `LinesearchMethod` is usually used in [`Linesearch`](@ref) (or with [`solve`](@ref)).

It is a subtype of `SolverMethod` (imported from `GeometricBase`) — line searches
are one-dimensional subproblems used *inside* nonlinear solvers and
optimizers, so (unlike a [`NonlinearSolverMethod`](@ref)) a `LinesearchMethod` is
not itself a nonlinear-solver method.
"""
abstract type LinesearchMethod{T} <: SolverMethod end

Base.eltype(::LinesearchMethod{T}) where {T} = T

"""
    change_precision(T, method::LinesearchMethod)

Return a copy of the [`LinesearchMethod`](@ref) `method` with its numeric fields
converted to the element type `T`.

This is an internal helper used when constructing a [`Linesearch`](@ref): the
method's precision is adapted to the working precision `T`.  It replaces a former
misuse of `Base.convert` (which was ambiguous with `Base` and violated the
`convert` contract by returning a differently-typed object).
"""
change_precision(::Type{T}, method::LinesearchMethod{T}) where {T} = method
change_precision(::Type, method::LinesearchMethod) =
    error("change_precision not implemented for $(typeof(method)).")


"""
    Linesearch

A `struct` that stores a [`LinesearchProblem`](@ref), [`LinesearchMethod`](@ref) and [`Options`](@ref).

# Keys

- `problem::`[`LinesearchProblem`](@ref)
- `method::`[`LinesearchMethod`](@ref)
- `config::`[`Options`](@ref)

# Constructors

The following constructors can be used:

```julia
Linesearch{T}(problem, method, config)
Linesearch(problem, method=Static(); kwargs...)
Linesearch(problem, method, config::Options)
```
"""
struct Linesearch{T,MET<:LinesearchMethod{T},PT<:LinesearchProblem{T},OPT<:Options{T}}
    problem::PT
    method::MET
    config::OPT
    Linesearch{T}(problem, method, config) where {T} = new{T,typeof(method),typeof(problem),typeof(config)}(problem, method, config)
end

Linesearch(problem::LinesearchProblem{T}, method::LinesearchMethod=Static(); options_kwargs...) where {T} =
    Linesearch{T}(problem, change_precision(T, method), Options(T; options_kwargs...))

Linesearch(problem::LinesearchProblem{T}, method::LinesearchMethod, config::Options{T}) where {T} =
    Linesearch{T}(problem, change_precision(T, method), config)

function solve(prob::LinesearchProblem{T}, method::LinesearchMethod, α, params=NullParameters(), config::Options{T}=Options(T)) where {T}
    solve(Linesearch(prob, method, config), T(α), params)
end

problem(s::Linesearch) = s.problem
config(s::Linesearch) = s.config
method(s::Linesearch) = s.method


"""
    solve(linesearch, α, params=NullParameters())

Solve the [`LinesearchProblem`](@ref) (contained in [`Linesearch`](@ref)) starting at `α`.

The argument `params` needs to be of an appropriate form expected by the respective [`LinesearchProblem`](@ref).

See [`linesearch_problem`](@ref).
"""
function solve(::Linesearch{T,MET}, α, params=NullParameters()) where {T, MET<:LinesearchMethod{T}}
    error("Solve method missing for $(MET).")
end
