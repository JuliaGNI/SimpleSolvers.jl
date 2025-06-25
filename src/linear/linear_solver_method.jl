"""
    LinearSolverMethod <: SolverMethod

Summarizes all the methods used for solving *linear systems of equations* such as the [`LU`](@ref) method.

# Extended help

The abstract type `SolverMethod` was imported from `GeometricBase`.
"""
abstract type LinearSolverMethod <: SolverMethod end

"""
    solve(ls, method)

Solve the [`LinearSystem`](@ref) `ls` with the [`LinearSolverMethod`](@ref) `method`.
"""
function solve(::LinearSystem, method::LinearSolverMethod) 
    error("Method solve not implemented for $(typeof(method)).")
end