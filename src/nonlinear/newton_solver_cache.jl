"""
    NewtonSolverCache

Stores `x̄`, `x`, `δx`, `rhs`, `y` and `J`.

Compare this to [`NewtonOptimizerCache`](@ref).

# Keys

- `x̄`: the previous iterate,
- `x`: the next iterate (or *guess* thereof). The *guess* is computed when calling the functions created by [`linesearch_problem`](@ref),
- `δx`: search direction. This is updated when calling [`solver_step!`](@ref) via the [`LinearSolver`](@ref) stored in the [`NewtonSolver`](@ref),
- `rhs`: the right-hand-side (this can be accessed by calling [`rhs`](@ref)), 
- `y`: the problem evaluated at `x`. This is used in [`linesearch_problem`](@ref),
- `J::AbstractMatrix`: the Jacobian evaluated at `x`. This is used in [`linesearch_problem`](@ref). Note that this is not of type [`Jacobian`](@ref)!

# Constructor

```julia
NewtonSolverCache(x, y)
```
"""
struct NewtonSolverCache{T, AT <: AbstractVector{T}, JT <: AbstractMatrix{T}} <: NonlinearSolverCache{T} # , JT <: AbstractMatrix{T}}
    x̄::AT
    x::AT
    δx::AT

    rhs::AT
    y::AT

    j::JT

    function NewtonSolverCache(x::AT, y::AT) where {T,AT<:AbstractArray{T}}
        j = alloc_j(x, y)
        c = new{T, AT, typeof(j)}(zero(x), zero(x), zero(x), zero(y), zero(y), j)
        initialize!(c, fill!(similar(x), NaN))
        c
    end
end

# jacobian(cache::NewtonSolverCache) = cache.J
direction(cache::NewtonSolverCache) = cache.δx
jacobian(cache::NewtonSolverCache) = cache.j

value(cache::NewtonSolverCache) = cache.y

@doc raw"""
    update!(cache, x)

Update the [`NewtonSolverCache`](@ref) based on `x`, i.e.:
1. `cache.x̄` ``\gets`` x,
2. `cache.x` ``\gets`` x,
3. `cache.δx` ``\gets`` 0.
"""
function update!(cache::NewtonSolverCache{T}, x::AbstractVector{T}) where {T}
    cache.x̄ .= x
    solution(cache) .= x
    cache.δx .= 0
    
    cache
end

"""
    initialize!(cache, x)

Initialize the [`NewtonSolverCache`](@ref) based on `x`.

# Implementation

This calls [`alloc_x`](@ref) to do all the initialization.
"""
function initialize!(cache::NewtonSolverCache{T}, ::AbstractVector{T}) where {T}
    cache.x̄ .= T(NaN)
    solution(cache) .= T(NaN)
    cache.δx .= T(NaN)

    cache.rhs .= T(NaN)
    cache.y .= T(NaN)

    jacobian(cache) .= T(NaN)

    cache
end

solution(cache::NewtonSolverCache) = cache.x

"""
    rhs(cache)

Return the right-hand side of the equation, stored in `cache::`[`NewtonSolverCache`](@ref).
"""
rhs(cache::NewtonSolverCache) = cache.rhs