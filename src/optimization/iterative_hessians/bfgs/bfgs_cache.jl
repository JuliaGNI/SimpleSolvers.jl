"""
    BFGSCache

The [`OptimizerCache`](@ref) for the [`BFGS`](@ref) algorithm. Also see [`update!(::BFGSCache, ::OptimizerState, ::AbstractVector, ::AbstractVector`)](@ref).
"""
struct BFGSCache{T, VT, MT} <: OptimizerCache{T} 
    x::VT    # current solution

    g::VT    # current gradient

    T1::MT
    T2::MT
    T3::MT
    δγ::MT
    δδ::MT

    rhs::VT
    Δx::VT
    Δg::VT

    function BFGSCache(x::AT) where {T, AT <: AbstractVector{T}}
        q = zeros(T, length(x), length(x))
        cache = new{T, AT, typeof(q)}(similar(x), similar(x), similar(q), similar(q), similar(q), similar(q), similar(q), similar(x), similar(x), similar(x))
        initialize!(cache, x)
        cache
    end
end

OptimizerCache(::BFGS, x::AbstractVector) = BFGSCache(x)

"""
    rhs(cache)

Return the right hand side of an instance of [`BFGSCache`](@ref)
"""
rhs(cache::BFGSCache) = cache.rhs

"""
    gradient(cache)

Return the stored gradient (array) of an instance of [`BFGSCache`](@ref)
"""
gradient(cache::BFGSCache) = cache.g

"""
    direction(cache)

Return the direction of the gradient step (i.e. `δ`) of an instance of [`BFGSCache`](@ref).
"""
direction(cache::BFGSCache) = cache.Δx

hessian(::BFGSCache) = error("BFGSCache does not store the Hessian, but it's inverse! Call inverse_hessian.")
inverse_hessian(::BFGSCache) = error("The inverse Hessian is stored in the state, not the cache!")

function update!(cache::BFGSCache, state::OptimizerState, x::AbstractVector)
    cache.x .= x
    direction(cache) .= cache.x - state.x̄
    outer!(cache.δδ, direction(cache), direction(cache))
    cache
end

@doc raw"""
    update!(cache, x, g)

Update the [`BFGSCache`](@ref) based on `x` and `g`.

# Extended help

The update rule used here can be found in [kochenderfer2019algorithms](@cite) and [nocedal2006numerical](@cite):

What this is doing is:

```math
\begin{aligned}
\delta & \gets x^{(k)} - x^{(k-1)}, \\
\gamma & \gets \nabla{}f^{(k)} - \nabla{}f^{(k-1)}, \\
T_1 & \gets \delta\gamma^TQ, \\
T_2 & \gets Q\gamma\delta^T, \\
T_3 & \gets (1 + \frac{gamma^TQ\gamma}{\delta^\gamma})\delta\delta^T,
Q & \gets Q - (T_1 + T_2 - T_3)/{\delta^T\gamma}
\end{aligned}
```
"""
function update!(cache::BFGSCache{T}, state::BFGSState{T}, x::AbstractVector{T}, g::AbstractVector{T}) where {T}
    update!(cache, state, x)
    gradient(cache) .= g
    rhs(cache) .= -g
    cache.Δx .= cache.x - state.x̄
    cache.Δg .= gradient(cache) - state.ḡ

    δγ = cache.Δx ⋅ cache.Δg

    if !iszero(δγ)
        outer!(cache.δδ, cache.Δx, cache.Δx)
        outer!(cache.δγ, cache.Δx, cache.Δg)
        mul!(cache.T1, cache.δγ, state.Q)
        mul!(cache.T2, state.Q, cache.δγ')
        γQγ = cache.Δg' * state.Q * cache.Δg
        cache.T3 .= (one(T) + γQγ ./ δγ) .* cache.δδ
        inverse_hessian(state) .-= (cache.T1 .+ cache.T2 .- cache.T3) ./ δγ
    end

    direction(cache) .= inverse_hessian(state) * rhs(cache)

    cache
end

update!(cache::BFGSCache, state::OptimizerState, grad::Gradient, x::AbstractVector) = update!(cache, state, x, grad(x))

update!(cache::BFGSCache, state::OptimizerState, grad::Gradient, ::HessianBFGS, x::AbstractVector) = update!(cache, state, grad, x)

function initialize!(cache::BFGSCache{T}, ::AbstractVector{T}) where {T}
    cache.x .= T(NaN)
    direction(cache) .= T(NaN)
    cache.g .= T(NaN)
    cache.rhs .= T(NaN)
    cache
end