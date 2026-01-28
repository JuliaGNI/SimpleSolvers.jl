"""
    LinearSolverMethod <: SolverMethod

Summarizes all the methods used for solving *linear systems of equations* such as the [`LU`](@ref) method.

# Extended help

The abstract type `SolverMethod` was imported from `GeometricBase`.
"""
abstract type LinearSolverMethod <: SolverMethod end

abstract type DirectMethod <: LinearSolverMethod end
# abstract type IterativeMethod <: LinearSolverMethod end

"""
    solve(ls, method)

Solve the [`LinearProblem`](@ref) `ls` with the [`LinearSolverMethod`](@ref) `method`.
"""
function solve(::LinearProblem, method::LinearSolverMethod) 
    error("Method solve not implemented for $(typeof(method)).")
end