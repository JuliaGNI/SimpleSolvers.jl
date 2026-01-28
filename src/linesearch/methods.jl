"""
    LinesearchMethod

Examples include [`Static`](@ref), [`Backtracking`](@ref), [`Bisection`](@ref) and [`Quadratic`](@ref).
See these examples for specific information on linesearch algorithms.

# Extended help

A `LinesearchMethod` always has to be used together in [`Linesearch`](@ref) (or with [`solve`](@ref)).
"""
abstract type LinesearchMethod{T} <: NonlinearMethod end

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