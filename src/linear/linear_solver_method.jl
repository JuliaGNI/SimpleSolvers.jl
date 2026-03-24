"""
    LinearSolverMethod <: SolverMethod

Summarizes all the methods used for solving *linear systems of equations* such as the [`LU`](@ref) method.

# Extended help

The abstract type `SolverMethod` was imported from `GeometricBase`.
"""
abstract type LinearSolverMethod <: SolverMethod end

abstract type DirectMethod <: LinearSolverMethod end
# abstract type IterativeMethod <: LinearSolverMethod end
