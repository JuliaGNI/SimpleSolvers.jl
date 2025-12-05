"""
    IterativeHessian <: Hessian

An abstract type derived from [`Hessian`](@ref).
Its main purpose is defining a supertype that encompasses [`HessianBFGS`](@ref) and [`HessianDFP`](@ref) for dispatch.
"""
abstract type IterativeHessian{T} <: Hessian{T} end

problem(H::IterativeHessian) = H.problem
solution(H::IterativeHessian) = H.x
gradient(H::IterativeHessian) = H.g
direction(H::IterativeHessian) = H.δ

"""
    update!(H, x, g)

Update the [`IterativeHessian`](@ref) based on `x` and `g`.

Note that this [`update!`](@ref) method performs different operations from `update!(::NewtonOptimizerCache, ::AbstractVector, ::AbstractVector)`](@ref).
"""
function update!(H::IterativeHessian, x::AbstractVector, g::AbstractVector)
    H.ḡ .= gradient(H)
    H.x̄ .= solution(H)
    solution(H) .= x
    gradient(H) .= g
    
    # δ = x - x̄
    H.δ .= solution(H) .- H.x̄

    # γ = g - ḡ
    H.γ .= gradient(H) .- H.ḡ

    H
end

@doc raw"""
    compute_δQδ(H)

Compute `δQδ`, which is used to check whether ``Q = \mathrm{inv}(H)`` in [`HessianDFP`](@ref) is updated (similarly to [`compute_δγ`](@ref)) and used explicitly in [`HessianBFGS`](@ref).
"""
function compute_γQγ(H::IterativeHessian)
    dot(H.γ, H.Q, H.γ)
end

@doc raw"""
    compute_δγ(H)

Compute `δγ` which is used to check whether ``Q = \mathrm{inv}(H)`` is updated.
"""
function compute_δγ(H::IterativeHessian)
    direction(H) ⋅ H.γ
end

@doc raw"""
    compute_outpuer_products(H)

Compute the outer products ``\gamma\gamma^T`` and ``\delta\delta`` for the [`IterativeHessian`](@ref) `H`.

Note that this is in-place; all the results are stored in `H`.
"""
function compute_outer_products!(::HT) where {HT <: IterativeHessian}
    error("Method not implemented for $(HT).")
end

function (∇²f::IterativeHessian)(H::AbstractMatrix, x::AbstractVector; gradient = GradientAutodiff{T}(H.problem.F, length(x)))
    update!(∇²f, x; gradient = gradient)
    H .= inv(∇²f)
end

function (∇²f::IterativeHessian)(x::AbstractVector; gradient = GradientAutodiff{T}(H.problem.F, length(x)))
    H = alloc_h(x)
    ∇²f(H, x; gradient = gradient)
end