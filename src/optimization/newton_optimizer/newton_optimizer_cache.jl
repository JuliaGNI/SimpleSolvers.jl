"""
    NewtonOptimizerCache

# Keys

- `x`: current iterate (this stores the guess called by the functions generated with [`linesearch_problem`](@ref)),
- `δ`: direction of optimization step (difference between `x` and `x̄`); this is obtained by multiplying `rhs` with the inverse of the Hessian,
- `g`: gradient value (this stores the gradient associated with `x` called by the *derivative part* of [`linesearch_problem`](@ref)),
- `rhs`: the right hand side used to compute the update.

To understand how these are used in practice see e.g. [`linesearch_problem`](@ref).

Also compare this to [`NewtonSolverCache`](@ref).
"""
struct NewtonOptimizerCache{T, AT <: AbstractArray{T}, HT <: AbstractMatrix{T}} <: OptimizerCache{T}
    x::AT
    δ::AT
    g::AT
    rhs::AT
    Δx::AT
    Δg::AT
    H::HT

    function NewtonOptimizerCache(x::AT) where {T, AT <: AbstractArray{T}}
        h = zeros(T, length(x), length(x))
        cache = new{T, AT, typeof(h)}(similar(x), similar(x), similar(x), similar(x), similar(x), similar(x), h)
        initialize!(cache, x)
        cache
    end

    # we probably don't need this constructor
    function NewtonOptimizerCache(x::AT, problem::OptimizerProblem) where {T <: Number, AT <: AbstractArray{T}}
        g = Gradient(problem)(x)
        h = Hessian(problem)(x)
        new{T, AT, typeof(h)}(copy(x), copy(x), zero(x), g, -g, zero(x), zero(x), h)
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

hessian(cache::NewtonOptimizerCache) = cache.H

function update!(cache::NewtonOptimizerCache, state::OptimizerState, x::AbstractVector)
    cache.x .= x
    direction(cache) .= cache.x - state.x̄
    cache
end

"""
    update!(cache, x, g)

Update the [`NewtonOptimizerCache`](@ref) based on `x` and `g`.
"""
function update!(cache::NewtonOptimizerCache, state::OptimizerState, x::AbstractVector, g::AbstractVector)
    update!(cache, state, x)
    gradient(cache) .= g
    rhs(cache) .= -g
    cache
end

update!(cache::NewtonOptimizerCache, state::OptimizerState, grad::Gradient, x::AbstractVector) = update!(cache, state, x, grad(x))

@doc raw"""
    update!(cache::NewtonOptimizerCache, x, g, hes)

Update an instance of [`NewtonOptimizerCache`](@ref) based on `x`.

This is used in [`update!(::OptimizerState, ::AbstractVector)`](@ref).

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
"""
function update!(cache::NewtonOptimizerCache, state::OptimizerState, g::Gradient, ∇²f::Hessian, x::AbstractVector)
    update!(cache, state, g, x)
    ∇²f(hessian(cache), x)
    direction(cache) .= hessian(cache) \ rhs(cache)
    cache
end

function update!(cache::NewtonOptimizerCache, state::OptimizerState, g::Gradient, ∇²f::IterativeHessian, x::AbstractVector)
    update!(cache, state, g, x)
    ∇²f(hessian(cache), x; gradient = g)
    direction(cache) .= hessian(cache) * rhs(cache)
    cache
end

function initialize!(cache::NewtonOptimizerCache{T}, ::AbstractVector{T}) where {T}
    cache.x .= T(NaN)
    direction(cache) .= T(NaN)
    cache.g .= T(NaN)
    cache.rhs .= T(NaN)
    hessian(cache) .= T(NaN)
    cache
end