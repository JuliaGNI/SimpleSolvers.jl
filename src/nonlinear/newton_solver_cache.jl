"""
    NewtonSolverCache

Stores `x₀`, `x₁`, `δx`, `rhs`, `y` and `J`.

# Keys

- `x₀`: the previous iterate,
- `x₁`: the next iterate (or *guess* thereof). The *guess* is computed when calling the functions created by [`linesearch_objective`](@ref),
- `δx`: search direction,
- `rhs`: the right-hand-side, 
- `y`: the objective evaluated at `x₁`. This is used in [`linesearch_objective`](@ref),
- `J::AbstractMatrix`: the Jacobian evaluated at `x₁`. This is used in [`linesearch_objective`](@ref). Note that this is not of type [`Jacobian`](@ref)!

# Constructor

```julia
NewtonSolverCache(x, y)
```

`J` is allocated by calling [`alloc_j`](@ref).

Also compare this to [`NewtonOptimizerCache`](@ref).
"""
struct NewtonSolverCache{T, AT <: AbstractVector{T}, JT <: AbstractMatrix{T}}
    x₀::AT
    x₁::AT
    δx::AT

    rhs::AT
    y::AT
    J::JT

    function NewtonSolverCache(x::AT, y::AT) where {T,AT<:AbstractArray{T}}
        J = alloc_j(x, y)
        c = new{T,AT,typeof(J)}(zero(x), zero(x), zero(x), zero(y), zero(y), J)
        initialize!(c, fill!(similar(x), NaN))
        c
    end
end

jacobian(cache::NewtonSolverCache) = cache.J
direction(cache::NewtonSolverCache) = cache.δx

@doc raw"""
    update!(cache, x)

Update the [`NewtonSolverCache`](@ref) based on `x`, i.e.:
1. `cache.x₀` ``\gets`` x,
2. `cache.x₁` ``\gets`` x,
3. `cache.δx` ``\gets`` 0.
"""
function update!(cache::NewtonSolverCache, x::AbstractVector)
    cache.x₀ .= x
    cache.x₁ .= x
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
    cache.x₀ .= alloc_x(x)
    cache.x₁ .= alloc_x(x)
    cache.δx .= alloc_x(x)

    cache.rhs .= alloc_x(x)
    cache.y .= alloc_f(cache.y)
    cache.J .= alloc_j(x, cache.y)

    cache
end