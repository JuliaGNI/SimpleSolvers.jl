"""
    DogLegCache

Like [`NonlinearSolverCache`](@ref) but storing two directions (callable with [`direction₁`](@ref) and [`direction₂`](@ref)).
"""
struct DogLegCache{T,AT<:AbstractVector{T},JT<:AbstractMatrix{T}} <: AbstractNonlinearSolverCache{T}
    x::AT
    Δx₁::AT
    Δx₂::AT
    Δx::AT
    Δx_diff::AT

    rhs::AT
    y::AT

    j::JT

    function DogLegCache(x::AT, y::AT) where {T,AT<:AbstractArray{T}}
        j = alloc_j(x, y)
        c = new{T,AT,typeof(j)}(zero(x), zero(x), zero(x), zero(x), zero(x), zero(y), zero(y), j)
        initialize!(c, fill!(similar(x), NaN))
        c
    end
end

"""
    direction₁(cache::DogLegCache)

Return the Newton direction.
"""
direction₁(cache::DogLegCache) = cache.Δx₁
"""
    direction₂(cache::DogLegCache)

Return the Gauss-Newton direction.
"""
direction₂(cache::DogLegCache) = cache.Δx₂

direction(cache::DogLegCache) = cache.Δx # error("The DoglegSolver stores two directions -> try `direction₁` or `direction₂`.")
direction_difference(cache::DogLegCache) = cache.Δx_diff
jacobianmatrix(cache::DogLegCache) = cache.j
solution(cache::DogLegCache) = cache.x
value(cache::DogLegCache) = cache.y
rhs(cache::DogLegCache) = cache.rhs

function initialize!(cache::DogLegCache{T}, ::AbstractVector{T}) where {T}
    solution(cache) .= T(NaN)
    direction₁(cache) .= T(NaN)
    direction₂(cache) .= T(NaN)
    direction(cache) .= T(NaN)
    direction_difference(cache) .= T(NaN)

    rhs(cache) .= T(NaN)
    value(cache) .= T(NaN)

    jacobianmatrix(cache) .= T(NaN)

    cache
end
