
struct HessianDFP{T} <: HessianParameters{T}
    δ::Vector{T}
    γ::Vector{T}

    Q::Matrix{T}

    T1::Matrix{T}
    T2::Matrix{T}
    γγ::Matrix{T}
    δδ::Matrix{T}

    function HessianDFP(x::Vector{T}) where {T}
        Q = zeros(T, length(x), length(x))

        T1 = zero(Q)
        T2 = zero(Q)
        γγ = zero(Q)
        δδ = zero(Q)
            
        new{T}(zero(x), zero(x), Q, T1, T2, γγ, δδ)
    end
end

HessianDFP(F, x) = HessianDFP(x)

initialize!(H::HessianDFP, ::Any) = H.Q .= Matrix(1.0I, size(H.Q)...)

function update!(H::HessianDFP, status::OptimizerStatus)
    # δ = x - x̄
    H.δ .= status.x .- status.x̄

    # γ = g - ḡ
    H.γ .= status.g .- status.ḡ

    # γQγ = γᵀQγ
    γQγ = dot(H.γ, H.Q, H.γ)

    # δγ = δᵀγ
    δγ = status.δ ⋅ status.γ

    # DFP
    # Q = Q - ... + ...
    # H.Q .-= H.Q * H.γ * H.γ' * H.Q / (H.γ' * H.Q * H.γ) .-
    #         H.δ * H.δ' ./ δγ

    outer!(H.γγ, H.γ, H.γ)
    outer!(H.δδ, H.δ, H.δ)

    mul!(H.T1, H.γγ, H.Q)
    mul!(H.T2, H.Q, H.T1)

    H.Q .-= H.T2 ./ γQγ
    H.Q .+= H.δδ ./ δγ
end

Base.inv(H::HessianDFP) = H.Q

Base.:\(H::HessianDFP, b) = inv(H) * b

LinearAlgebra.ldiv!(x, H::HessianDFP, b) = mul!(x, inv(H), b)
