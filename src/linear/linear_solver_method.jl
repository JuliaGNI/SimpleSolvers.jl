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
    PivotedLUMethod <: DirectMethod

The methods that compute a partially-pivoted LU factorization in LAPACK's layout, and can
therefore share a cache, a triangular solve and everything built on them: [`LapackLU`](@ref)
and [`RecursiveLU`](@ref).

They differ only in which kernel computes the factors — see [`_getrf!`](@ref) — and in which
element types they accept. See [`PivotedLUCache`](@ref).
"""
abstract type PivotedLUMethod <: DirectMethod end

"""
    SparseDirectMethod <: DirectMethod

The direct methods that factorize a *sparse* matrix, keeping the ordering and symbolic
factorization across refactorizations: [`UmfpackLU`](@ref) and [`SparspakLU`](@ref).

Unlike a [`PivotedLUMethod`](@ref) these need the sparsity pattern up front, at
[`LinearSolver`](@ref) construction, and they refuse a dense matrix. See
[`SparseFactorizationCache`](@ref).
"""
abstract type SparseDirectMethod <: DirectMethod end
