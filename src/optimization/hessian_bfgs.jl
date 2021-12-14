
struct HessianBFGS{T} <: HessianParameters{T}
    δ::Vector{T}
    γ::Vector{T}

    Q::Matrix{T}

    T1::Matrix{T}
    T2::Matrix{T}
    T3::Matrix{T}
    δγ::Matrix{T}
    δδ::Matrix{T}

    function HessianBFGS(x::Vector{T}) where {T}
        Q  = zeros(T, length(x), length(x))

        T1 = zero(Q)
        T2 = zero(Q)
        T3 = zero(Q)
        δγ = zero(Q)
        δδ = zero(Q)
            
        new{T}(zero(x), zero(x), Q, T1, T2, T3, δγ, δδ)
    end
end

HessianBFGS(F, x) = HessianBFGS(x)

initialize!(H::HessianBFGS, ::Any) = H.Q .= Matrix(1.0I, size(H.Q)...)

function update!(H::HessianBFGS, status::OptimizerStatus)
    # δ = x - x̄
    H.δ .= status.x .- status.x̄

    # γ = g - ḡ
    H.γ .= status.g .- status.ḡ

    # δγ = δᵀγ
    δγ = status.δ ⋅ status.γ

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
