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
    y₂::AT
    y₃::AT

    j::JT

    # Trust-region radius, carried across outer solver steps (see [`solver_step!`]
    # for the [`DogLegSolver`](@ref)).  A `Ref` so it can be mutated in place while
    # the surrounding cache stays immutable.
    Δ::Base.RefValue{T}

    # `j` is a prototype, and copied; see the `NonlinearSolverCache` counterpart for both.
    function DogLegCache(x::AT, y::AT, jprototype::AbstractMatrix{T}=alloc_j(x, y)) where {T,AT<:AbstractVector{T}}
        j = copy(jprototype)
        c = new{T,AT,typeof(j)}(zero(x), zero(x), zero(x), zero(x), zero(x), zero(y), zero(y), zero(y), zero(y), j, Ref(T(DOGLEG_Δ_INITIAL)))
        initialize!(c, fill!(similar(x), NaN))
        c
    end
end

"""
    trust_radius(cache::DogLegCache)

Return the current trust-region radius ``\\Delta`` carried by the [`DogLegCache`](@ref).
"""
trust_radius(cache::DogLegCache) = cache.Δ[]

"""
    trust_radius!(cache::DogLegCache, Δ)

Store the trust-region radius ``\\Delta`` in the [`DogLegCache`](@ref) so it carries
over to the next outer solver step.
"""
trust_radius!(cache::DogLegCache{T}, Δ::T) where {T} = (cache.Δ[] = Δ)

"""
    direction₁(cache::DogLegCache)

Return the steepest descent direction.

See [`directions!`](@ref).
"""
direction₁(cache::DogLegCache) = cache.Δx₁
"""
    direction₂(cache::DogLegCache)

Return the Newton direction.

See [`directions!`](@ref).
"""
direction₂(cache::DogLegCache) = cache.Δx₂

direction(cache::DogLegCache) = cache.Δx
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
    cache.y₂ .= T(NaN)
    cache.y₃ .= T(NaN)

    fill_nan!(jacobianmatrix(cache))

    # Reset the trust-region radius: it is carried *across solver steps within one
    # solve*, but a fresh solve (solver reuse) must not inherit the radius the
    # previous solve ended with.
    trust_radius!(cache, T(DOGLEG_Δ_INITIAL))

    cache
end
