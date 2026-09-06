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
    RankRevealingMethod{RT} <: DirectMethod

The direct methods that determine a *numerical rank* while they factorize, and solve a
rank-deficient system with the minimum-norm solution instead of raising a
`SingularException`: [`PivotedQR`](@ref) and [`SVDSolver`](@ref).

They differ in the decomposition — a complete orthogonal factorization against a singular
value decomposition — and therefore in cost and in what they can report, but they share the
rank tolerance, [`rank`](@ref), [`singular_index`](@ref) and every [`solve!`](@ref) form. See
[`rank_tolerance`](@ref).

A [`PivotedLUMethod`](@ref) is the right choice whenever a singular matrix would be a bug;
these are for the case where it is a property of the problem. Neither is ever selected by
[`default_linear_solver_method`](@ref) — see its docstring for why.

The type parameter `RT` is the type of the `rtol` field: `Missing` for a method that resolves
its tolerance from the element type it is handed, and a `Real` for one given an explicit
tolerance. It is a parameter of the abstract type because [`rank_tolerance`](@ref) dispatches
on it once for both methods.
"""
abstract type RankRevealingMethod{RT <: Union{Missing, Real}} <: DirectMethod end

"""
    SparseDirectMethod <: DirectMethod

The direct methods that factorize a *sparse* matrix, keeping the ordering and symbolic
factorization across refactorizations: [`UmfpackLU`](@ref) and [`SparspakLU`](@ref).

Unlike a [`PivotedLUMethod`](@ref) these need the sparsity pattern up front, at
[`LinearSolver`](@ref) construction, and they refuse a dense matrix. See
[`SparseFactorizationCache`](@ref).
"""
abstract type SparseDirectMethod <: DirectMethod end
