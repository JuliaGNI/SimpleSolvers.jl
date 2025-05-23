"""
    LinearSolverCache

An abstract type that summarizes all the caches used for [`LinearSolver`](@ref)s. See e.g. [`LUSolverCache`](@ref).
"""
abstract type LinearSolverCache{T} end

LinearSolverCache(method::LinearSolverMethod, ::AbstractArray) = error("No LinearSolverCache method implemented for method $(typeof(method)).")