"""
    NewtonSolverCache

Stores `x̄`, `x`, `δx`, `rhs`, `y` and `J`.

Compare this to [`NewtonOptimizerCache`](@ref).

# Keys

- `x̄`: the previous iterate,
- `x`: the next iterate (or *guess* thereof). The *guess* is computed when calling the functions created by [`linesearch_objective`](@ref),
- `δx`: search direction. This is updated when calling [`solver_step!`](@ref) via the [`LinearSolver`](@ref) stored in the [`NewtonSolver`](@ref),
- `rhs`: the right-hand-side (this can be accessed by calling [`rhs`](@ref)), 
- `y`: the objective evaluated at `x`. This is used in [`linesearch_objective`](@ref),
- `J::AbstractMatrix`: the Jacobian evaluated at `x`. This is used in [`linesearch_objective`](@ref). Note that this is not of type [`Jacobian`](@ref)!

# Constructor

```julia
NewtonSolverCache(x, y)
```

# Implementation

`J` is allocated by calling [`alloc_j`](@ref).
"""
struct NewtonSolverCache{T, AT <: AbstractVector{T}} # , JT <: AbstractMatrix{T}}
    x̄::AT
    x::AT
    δx::AT

    rhs::AT
    y::AT
    # J::JT

    function NewtonSolverCache(x::AT, y::AT) where {T,AT<:AbstractArray{T}}
        # J = alloc_j(x, y)
        # c = new{T,AT,typeof(J)}(zero(x), zero(x), zero(x), zero(y), zero(y), J)
        c = new{T, AT}(zero(x), zero(x), zero(x), zero(y), zero(y))
        initialize!(c, fill!(similar(x), NaN))
        c
    end
end

# jacobian(cache::NewtonSolverCache) = cache.J
direction(cache::NewtonSolverCache) = cache.δx

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
function initialize!(cache::NewtonSolverCache, x::AbstractVector)
    cache.x̄ .= alloc_x(x)
    solution(cache) .= alloc_x(x)
    cache.δx .= alloc_x(x)

    cache.rhs .= alloc_x(x)
    cache.y .= alloc_f(cache.y)
    # cache.J .= alloc_j(x, cache.y)

    cache
end

solution(cache::NewtonSolverCache) = cache.x
"""
    rhs(cache)

Return the right-hand side of the equation, stored in `cache::`[`NewtonSolverCache`](@ref).
"""
rhs(cache::NewtonSolverCache) = cache.rhs