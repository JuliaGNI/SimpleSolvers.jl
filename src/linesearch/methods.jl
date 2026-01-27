"""
    LinesearchMethod

Examples include [`StaticState`](@ref), [`Backtracking`](@ref), [`Bisection`](@ref) and [`Quadratic`](@ref).
See these examples for specific information on linesearch algorithms.

# Extended help

A `LinesearchMethod` always has to be used together in [`Linesearch`](@ref) (or with [`solve`](@ref)).
"""
abstract type LinesearchMethod <: NonlinearMethod end

function solve(ls_prob::LinesearchProblem, ls_method::LinesearchMethod; config::Options=Options())
    solve(ls_prob, Linesearch(ls_method, config))
end