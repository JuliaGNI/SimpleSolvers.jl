"""
    LinesearchMethod

Examples include [`Static`](@ref), [`Backtracking`](@ref), [`Bisection`](@ref) and [`Quadratic`](@ref).
See these examples for specific information on linesearch algorithms.

# Extended help

A `LinesearchMethod` always has to be used together in [`Linesearch`](@ref) (or with [`solve`](@ref)).
"""
abstract type LinesearchMethod{T} <: NonlinearMethod end

"""
    _linesearch_factor(ls::LinesearchMethod)

Returns the *factor* used in the linesearch algorithm.
This is used for checking if `NaN`s or `Inf`s can be expected.
"""
_linesearch_factor(::LinesearchMethod{T}) where {T} = one(T)

Base.eltype(::LinesearchMethod{T}) where {T} = T

function solve(ls_prob::LinesearchProblem, ls_method::LinesearchMethod; config::Options=Options())
    solve(ls_prob, Linesearch(ls_method, config))
end

function Base.convert(::Type, algorithm::LinesearchMethod)
    error("Convert not implemented for $(typeof(algorithm)).")
end

function Base.convert(::Type{T}, algorithm::LinesearchMethod{T}) where {T}
    algorithm
end
