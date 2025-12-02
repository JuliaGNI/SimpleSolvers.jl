"""
    HessianDFP <: Hessian


"""
struct HessianDFP{T,VT,MT,OBJ} <: IterativeHessian{T}
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
    γγ::MT
    δδ::MT

    function HessianDFP(problem::OptimizerProblem, x::AbstractVector{T}) where {T}
        Q = alloc_h(x)

        T1 = zero(Q)
        T2 = zero(Q)
        γγ = zero(Q)
        δδ = zero(Q)
            
        H = new{T,typeof(x),typeof(Q),typeof(problem)}(problem, zero(x), zero(x), zero(x), zero(x), zero(x), zero(x), Q, T1, T2, γγ, δδ)
        initialize!(H, x)
        H
    end
end


HessianDFP(F::Callable, x::AbstractVector) = HessianDFP(OptimizerProblem(F, x), x)

HessianDFP{T}(F::Callable, n::Integer) where {T} = HessianDFP(F, zeros(T, n))

Hessian(::DFP, ForOBJ::Union{Callable, OptimizerProblem}, x::AbstractVector) = HessianDFP(ForOBJ, x)

function initialize!(H::HessianDFP{T}, x::AbstractVector{T}) where {T}
    H.Q .= Matrix(one(T) * I, size(H.Q)...)

    H.x̄ .= eltype(x)(NaN)
    H.δ .= eltype(x)(NaN)
    H.ḡ .= eltype(x)(NaN)
    H.γ .= eltype(x)(NaN)

    H.x .= x
    grad = GradientAutodiff{T}(H.problem.F, length(x))
    H.g .= grad(H.problem, x)

    H
end

function compute_outer_products!(H::HessianDFP)
    outer!(H.γγ, H.γ, H.γ)
    outer!(H.δδ, H.δ, H.δ)
end

function update!(H::HessianDFP{T}, x::AbstractVector{T}) where {T}
    # copy previous data and compute new gradient
    H.ḡ .= H.g
    H.x̄ .= H.x
    H.x .= x
    grad = GradientAutodiff{T}(H.problem.F, length(x))
    H.g .= grad(H.problem, x)

    # δ = x - x̄
    direction(H) .= H.x - H.x̄

    # γ = g - ḡ
    H.γ .= H.g - H.ḡ

    # δγ = δ ⋅ γ
    δγ = compute_δγ(H)

    γQγ = compute_γQγ(H)

    # DFP
    # Q = Q - ... + ...
    # H.Q .-= H.Q * H.γ * H.γ' * H.Q / (H.γ' * H.Q * H.γ) .-
    #         H.δ * H.δ' ./ δγ

    if !iszero(δγ) && !iszero(γQγ)
        compute_outer_products!(H)

        mul!(H.T1, H.γγ, H.Q)
        mul!(H.T2, H.Q, H.T1)

        H.Q .-= H.T2 ./ γQγ
        H.Q .+= H.δδ ./ δγ
    end
end

Base.inv(H::HessianDFP) = H.Q

Base.:\(H::HessianDFP, b) = inv(H) * b