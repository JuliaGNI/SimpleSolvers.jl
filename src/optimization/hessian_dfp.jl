
struct HessianDFP{T,VT,MT,OBJ} <: Hessian{T}
    objective::OBJ

    x̄::VT    # previous solution
    x::VT    # current solution
    δ::VT    # difference of current and previous solution

    ḡ::VT    # previous gradient
    g::VT    # current gradient
    γ::VT    # difference of current and previous gradient

    Q::MT

    T1::MT
    T2::MT
    γγ::MT
    δδ::MT

    function HessianDFP(objective::MultivariateObjective, x::AbstractVector{T}) where {T}
        Q = alloc_h(x)

        T1 = zero(Q)
        T2 = zero(Q)
        γγ = zero(Q)
        δδ = zero(Q)
            
        new{T,typeof(x),typeof(Q),typeof(objective)}(objective, zero(x), zero(x), zero(x), zero(x), zero(x), zero(x), Q, T1, T2, γγ, δδ)
    end
end


HessianDFP(F, x) = HessianDFP(MultivariateObjective(F, x), x)

function initialize!(H::HessianDFP, x::AbstractVector)
    H.Q .= Matrix(1.0I, size(H.Q)...)

    H.x̄ .= eltype(x)(NaN)
    H.δ .= eltype(x)(NaN)
    H.ḡ .= eltype(x)(NaN)
    H.γ .= eltype(x)(NaN)

    H.x .= x
    H.g .= gradient!(H.objective, x)

    return H
end

function update!(H::HessianDFP, x::AbstractVector)
    # copy previous data and compute new gradient
    H.ḡ .= H.g
    H.x̄ .= H.x
    H.x .= x
    H.g .= gradient!(H.objective, x)

    # δ = x - x̄
    H.δ .= H.x .- H.x̄

    # γ = g - ḡ
    H.γ .= H.g .- H.ḡ

    # γQγ = γᵀQγ
    γQγ = dot(H.γ, H.Q, H.γ)

    # δγ = δᵀγ
    δγ = H.δ ⋅ H.γ

    # DFP
    # Q = Q - ... + ...
    # H.Q .-= H.Q * H.γ * H.γ' * H.Q / (H.γ' * H.Q * H.γ) .-
    #         H.δ * H.δ' ./ δγ

    if δγ ≠ 0 && γQγ ≠ 0
        outer!(H.γγ, H.γ, H.γ)
        outer!(H.δδ, H.δ, H.δ)

        mul!(H.T1, H.γγ, H.Q)
        mul!(H.T2, H.Q, H.T1)

        H.Q .-= H.T2 ./ γQγ
        H.Q .+= H.δδ ./ δγ
    end
end

Base.inv(H::HessianDFP) = H.Q

Base.:\(H::HessianDFP, b) = inv(H) * b

LinearAlgebra.ldiv!(x, H::HessianDFP, b) = mul!(x, inv(H), b)
