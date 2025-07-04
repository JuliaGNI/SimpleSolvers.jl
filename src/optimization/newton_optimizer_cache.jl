"""
    NewtonOptimizerCache

# Keys

- `x̄`: the previous iterate,
- `x`: current iterate (this stores the guess called by the functions generated with [`linesearch_objective`](@ref)),
- `δ`: direction of optimization step (difference between `x` and `x̄`); this is obtained by multiplying `rhs` with the inverse of the Hessian,
- `g`: gradient value (this stores the gradient associated with `x` called by the *derivative part* of [`linesearch_objective`](@ref)),
- `rhs`: the right hand side used to compute the update.

To understand how these are used in practice see e.g. [`linesearch_objective`](@ref).

Also compare this to [`NewtonSolverCache`](@ref).
"""
struct NewtonOptimizerCache{T, AT <: AbstractArray{T}}
    x̄::AT
    x::AT
    δ::AT
    g::AT
    rhs::AT

    function NewtonOptimizerCache(x::AT) where {T, AT <: AbstractArray{T}}
        cache = new{T,AT}(similar(x), similar(x), similar(x), similar(x), similar(x))
        initialize!(cache, x)
        cache
    end

    # we probably don't need this constructor
    function NewtonOptimizerCache(x::AT, objective::MultivariateObjective) where {T <: Number, AT <: AbstractArray{T}}
        g = gradient!(objective, x)
        new{T, AT}(copy(x), copy(x), zero(x), g, -g)
    end
end

"""
    rhs(cache)

Return the right hand side of an instance of [`NewtonOptimizerCache`](@ref)
"""
rhs(cache::NewtonOptimizerCache) = cache.rhs
"""
    gradient(::NewtonOptimizerCache)

Return the stored gradient (array) of an instance of [`NewtonOptimizerCache`](@ref)
"""
gradient(cache::NewtonOptimizerCache) = cache.g
"""
    direction(::NewtonOptimizerCache)

Return the direction of the gradient step (i.e. `δ`) of an instance of [`NewtonOptimizerCache`](@ref).
"""
direction(cache::NewtonOptimizerCache) = cache.δ

@doc raw"""
    update!(cache, x)

Update `cache` (an instance of [`NewtonOptimizerCache`](@ref)) based on `x`.

This does:
```math
\begin{aligned}
\bar{x}^\mathtt{cache} & \gets x, \\
x^\mathtt{cache} & \gets x, \\
\delta^\mathtt{cache} & \gets 0, 
\end{aligned}
```
where ``\delta^\mathtt{cache}`` is the [`direction`](@ref) stored in `cache`. 
"""
function update!(cache::NewtonOptimizerCache, x::AbstractVector)
    cache.x̄ .= x
    cache.x .= x
    direction(cache) .= 0
    cache
end

"""
    update!(cache, x, g)

Update the [`NewtonOptimizerCache`](@ref) based on `x` and `g`.
"""
function update!(cache::NewtonOptimizerCache, x::AbstractVector, g::AbstractVector)
    update!(cache, x)
    gradient(cache) .= g
    rhs(cache) .= -g
    cache
end

update!(cache::NewtonOptimizerCache, x::AbstractVector, g::Gradient) = update!(cache, x, gradient(x, g))

@doc raw"""
    update!(cache::NewtonOptimizerCache, x, g, hes)

Update an instance of [`NewtonOptimizerCache`](@ref) based on `x`.

This is used in [`update!(::NewtonOptimizerState, ::AbstractVector)`](@ref).

This sets:
```math
\bar{x}^\mathtt{cache} \gets x,
x^\mathtt{cache} \gets x,
g^\mathtt{cache} \gets g,
\mathrm{rhs}^\mathtt{cache} \gets -g,
\delta^\mathtt{cache} \gets H^{-1}\mathrm{rhs}^\mathtt{cache},
```
where we wrote ``H`` for the Hessian (i.e. the input argument `hes`). 

Also see [`update!(::NewtonSolverCache, ::AbstractVector)`](@ref). 

!!! warn
    Note that this is not updating the Hessian `hes`. For this call `update!` on the `Hessian`.

# Implementation

The multiplication by the inverse of ``H`` is done with `LinearAlgebra.ldiv!`.
"""
function update!(cache::NewtonOptimizerCache, x::AbstractVector, g::Union{AbstractVector, Gradient}, hes::Hessian)
    update!(cache, x, g)
    ldiv!(direction(cache), hes, rhs(cache))
    cache
end

function initialize!(cache::NewtonOptimizerCache{T}, x::AbstractVector{T}) where {T}
    cache.x̄ .= T(NaN)
    cache.x .= T(NaN)
    cache.δ .= T(NaN)
    cache.g .= T(NaN)
    cache.rhs .= T(NaN)
    cache
end
