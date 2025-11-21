"""
    HessianBFGS <: Hessian
"""
struct HessianBFGS{T,VT,MT,OBJ} <: Hessian{T}
    problem::OBJ

    x̄::VT    # previous solution
    x::VT    # current solution
    δ::VT    # difference of current and previous solution

    ḡ::VT    # previous gradient
    g::VT    # current gradient
    γ::VT    # difference of current and previous gradient

    Q::MT

    T1::MT
    T2::MT
    T3::MT
    δγ::MT
    δδ::MT

    function HessianBFGS(problem::OptimizerProblem, x::AbstractVector{T}) where {T}
        Q  = alloc_h(x)
        
        T1 = zero(Q)
        T2 = zero(Q)
        T3 = zero(Q)
        δγ = zero(Q)
        δδ = zero(Q)
            
        new{T,typeof(x),typeof(Q),typeof(problem)}(problem, zero(x), zero(x), zero(x), zero(x), zero(x), zero(x), Q, T1, T2, T3, δγ, δδ)
    end
end

HessianBFGS(F::Callable, x::AbstractVector) = HessianBFGS(OptimizerProblem(F, x), x)

Hessian(::BFGS, ForOBJ::Union{Callable, OptimizerProblem}, x::AbstractVector) = HessianBFGS(ForOBJ, x)

function initialize!(H::HessianBFGS{T}, x::AbstractVector{T}) where {T}
    H.Q .= Matrix(1.0I, size(H.Q)...)

    H.x̄ .= eltype(x)(NaN)
    H.δ .= eltype(x)(NaN)
    H.ḡ .= eltype(x)(NaN)
    H.γ .= eltype(x)(NaN)

    H.x .= x
    H.g .= gradient!(H.problem, GradientAutodiff{T}(H.problem.F, length(x)), x)

    H
end

function update!(H::HessianBFGS{T}, x::AbstractVector{T}) where {T}
    # copy previous data and compute new gradient
    H.ḡ .= H.g
    H.x̄ .= H.x
    H.x .= x
    H.g .= gradient!(H.problem, GradientAutodiff{T}(H.problem.F, length(x)), x)

    # δ = x - x̄
    H.δ .= H.x .- H.x̄

    # γ = g - ḡ
    H.γ .= H.g .- H.ḡ

    # δγ = δᵀγ
    δγ = H.δ ⋅ H.γ

    # BFGS
    # Q = Q - ... + ...
    # H.Q .-= (H.δ * H.γ' * H.Q .+ H.Q * H.γ * H.δ') ./ δγ .-
    #         (1 + dot(H.γ, H.Q, H.γ) ./ δγ) .* (H.δ * H.δ') ./ δγ

    if δγ ≠ 0
        outer!(H.δγ, H.δ, H.γ)
        outer!(H.δδ, H.δ, H.δ)
        mul!(H.T1, H.δγ, H.Q)
        mul!(H.T2, H.Q, H.δγ')
        H.T3 .= (1 + dot(H.γ, H.Q, H.γ) ./ δγ) .* H.δδ
        H.Q .-= (H.T1 .+ H.T2 .- H.T3) ./ δγ
    end
end

Base.inv(H::HessianBFGS) = H.Q

Base.:\(H::HessianBFGS, b) = inv(H) * b

LinearAlgebra.ldiv!(x, H::HessianBFGS, b) = mul!(x, inv(H), b)
