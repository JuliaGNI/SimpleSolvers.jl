"""
    AbstractNonlinearSolverCache

An abstract type that comprises e.g. the [`NonlinearSolverCache`](@ref).
"""
abstract type AbstractNonlinearSolverCache{T} end

"""
    NonlinearSolverCache

Stores `x`, `Δx`, `rhs`, `y`, and `J`.

Compare this to [`NewtonOptimizerCache`](@ref).

# Keys

- `x`: the next iterate (or *guess* thereof). The *guess* is computed when calling the functions created by [`linesearch_problem`](@ref),
- `Δx`: search direction. This is updated when calling [`solver_step!`](@ref) via the [`LinearSolver`](@ref) stored in the [`NewtonSolver`](@ref),
- `rhs`: the right-hand-side (this can be accessed by calling [`rhs`](@ref)),
- `y`: the problem evaluated at `x`. This is used in [`linesearch_problem`](@ref),
- `j::AbstractMatrix`: the Jacobian evaluated at `x`. This is used in [`linesearch_problem`](@ref). Note that this is not of type [`Jacobian`](@ref)!

# Constructor

```julia
NonlinearSolverCache(x, y)
```
"""
struct NonlinearSolverCache{T,AT<:AbstractVector{T},JT<:AbstractMatrix{T}} <: AbstractNonlinearSolverCache{T}
    x::AT
    Δx::AT

    rhs::AT
    y::AT

    j::JT

    function NonlinearSolverCache(x::AT, y::AT) where {T,AT<:AbstractArray{T}}
        j = alloc_j(x, y)
        c = new{T,AT,typeof(j)}(zero(x), zero(x), zero(y), zero(y), j)
        initialize!(c, fill!(similar(x), NaN))
        c
    end
end

direction(cache::NonlinearSolverCache) = cache.Δx
jacobianmatrix(cache::NonlinearSolverCache) = cache.j
solution(cache::NonlinearSolverCache) = cache.x
value(cache::NonlinearSolverCache) = cache.y

"""
    rhs(cache)

Return the right-hand side of the equation, stored in `cache::`[`NonlinearSolverCache`](@ref).
"""
rhs(cache::NonlinearSolverCache) = cache.rhs

"""
    initialize!(cache, x)

Initialize the [`NonlinearSolverCache`](@ref) based on `x`.

# Implementation

This calls [`alloc_x`](@ref) to do all the initialization.
"""
function initialize!(cache::NonlinearSolverCache{T}, ::AbstractVector{T}) where {T}
    solution(cache) .= T(NaN)
    direction(cache) .= T(NaN)

    rhs(cache) .= T(NaN)
    value(cache) .= T(NaN)

    jacobianmatrix(cache) .= T(NaN)

    cache
end
