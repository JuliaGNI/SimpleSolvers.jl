"""
    DFPCache <: OptimizerCache

The [`OptimizerCache`](@ref) corresponding to the [`DFP`](@ref) method.
"""
struct DFPCache{T,VT,MT} <: OptimizerCache{T}
    x::VT    # current solution

    g::VT    # current gradient

    T1::MT
    T2::MT
    ΔgΔg::MT
    ΔxΔx::MT

    rhs::VT
    Δx::VT
    Δg::VT

    function DFPCache(x::AT) where {T,AT<:AbstractVector{T}}
        q = zeros(T, length(x), length(x))
        cache = new{T,AT,typeof(q)}(similar(x), similar(x), similar(q), similar(q), similar(q), similar(q), similar(x), similar(x), similar(x))
        initialize!(cache, x)
        cache
    end
end

OptimizerCache(::DFP, x::AbstractVector) = DFPCache(x)

"""
    rhs(cache)

Return the right hand side of an instance of [`DFPCache`](@ref)
"""
rhs(cache::DFPCache) = cache.rhs

"""
    gradient(cache)

Return the stored gradient (array) of an instance of [`DFPCache`](@ref)
"""
gradient(cache::DFPCache) = cache.g

"""
    direction(cache)

Return the direction of the gradient step (i.e. `Δx`) of an instance of [`DFPCache`](@ref).
"""
direction(cache::DFPCache) = cache.Δx

solution(cache::DFPCache) = cache.x

hessian(::DFPCache) = error("DFPCache does not store the Hessian, but it's inverse! Call inverse_hessian.")
inverse_hessian(::DFPCache) = error("The inverse Hessian is stored in the state, not the cache!")

function update!(cache::DFPCache, state::OptimizerState, x::AbstractVector)
    cache.x .= x
    direction(cache) .= cache.x - state.x̄
    outer!(cache.ΔxΔx, direction(cache), direction(cache))
    cache
end

@doc raw"""
    update!(cache, x, g)

Update the [`DFPCache`](@ref) based on `x` and `g`.

# Extended help

The update rule used here can be found in [kochenderfer2019algorithms](@cite) and [nocedal2006numerical](@cite).
"""
function update!(cache::DFPCache{T}, state::DFPState{T}, x::AbstractVector{T}, g::AbstractVector{T}) where {T}
    update!(cache, state, x)
    gradient(cache) .= g
    rhs(cache) .= -g
    cache.Δx .= cache.x - state.x̄
    cache.Δg .= gradient(cache) - state.ḡ

    ΔxΔg = cache.Δx ⋅ cache.Δg
    γQγ = cache.Δg' * state.Q * cache.Δg

    if !iszero(ΔxΔg) & !iszero(γQγ)
        outer!(cache.ΔxΔx, cache.Δx, cache.Δx)
        outer!(cache.ΔgΔg, cache.Δg, cache.Δg)
        mul!(cache.T1, cache.ΔxΔx, state.Q)
        mul!(cache.T2, state.Q, cache.T1)
        state.Q .-= cache.T2 ./ γQγ
        state.Q .+= cache.ΔxΔx ./ ΔxΔg
    end

    direction(cache) .= inverse_hessian(state) * rhs(cache)

    cache
end

update!(cache::DFPCache, state::OptimizerState, grad::Gradient, x::AbstractVector) = update!(cache, state, x, grad(x))

update!(cache::DFPCache, state::OptimizerState, grad::Gradient, ::HessianDFP, x::AbstractVector) = update!(cache, state, grad, x)

function initialize!(cache::DFPCache{T}, ::AbstractVector{T}) where {T}
    cache.x .= T(NaN)
    direction(cache) .= T(NaN)
    cache.g .= T(NaN)
    cache.rhs .= T(NaN)
    cache
end
