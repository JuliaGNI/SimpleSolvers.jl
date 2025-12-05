"""
    HessianBFGS <: Hessian

A `struct` derived from [`Hessian`](@ref) to be used for an [`Optimizer`](@ref).

# Fields
- `problem::`[`OptimizerProblem`](@ref): 
- `x̄`: previous solution,
- `x`: current solution,
- `δ`: *descent direction*,
- `ḡ`: previous gradient,
- `g`: current gradient,
- `γ`: difference between current and previous gradient,
- `Q`: 
- `T1`:
- `T2`:
- `T3`:
- `δγ`: the outer product of `δ` and `γ`. Note that this is different from the output of [`compute_δγ`](@ref), which is the inner product of `γ` and `δ`.
- `δδ`: 

Also compare those fields with the ones of [`NewtonOptimizerCache`](@ref).
"""
struct HessianBFGS{T,VT,MT,FT <: Callable} <: IterativeHessian{T}
    F::FT

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

    function HessianBFGS(F::FT, x::AbstractVector{T}) where {T, FT <: Callable}
        Q  = alloc_h(x)
        
        T1 = zero(Q)
        T2 = zero(Q)
        T3 = zero(Q)
        δγ = zero(Q)
        δδ = zero(Q)
            
        H = new{T,typeof(x),typeof(Q),FT}(F, zero(x), zero(x), zero(x), zero(x), zero(x), zero(x), Q, T1, T2, T3, δγ, δδ)
        initialize!(H, x)
        H
    end
end

HessianBFGS{T}(F::Callable, n::Integer) where {T} = HessianBFGS(F, zeros(T, n))

HessianBFGS(obj::OptimizerProblem, x::AbstractVector) = HessianBFGS(obj.F, x)

Hessian(::BFGS, ForOBJ::Callable, x::AbstractVector) = HessianBFGS(ForOBJ, x)

Hessian(::BFGS, ForOBJ::OptimizerProblem, x::AbstractVector) = HessianBFGS(ForOBJ.F, x)

@doc raw"""
    initialize!(H, x)

Initialize an object `H` of type [`HessianBFGS`](@ref). 

We note that unlike most other [`initialize!`](@ref) methods this one is not writing `NaN`s everywhere - ``Q = H^{-1}`` is set to the identity.

# Examples

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: initialize!)
f(x) = sum(x .^ 2)
x = [1f0, 2f0, 3f0]
H = HessianBFGS(f, x)
initialize!(H, x)
inv(H)

# output

3×3 Matrix{Float32}:
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  1.0
```
"""
function initialize!(H::HessianBFGS{T}, x::AbstractVector{T}; gradient = GradientAutodiff{T}(H.F, length(x))) where {T}
    H.Q .= Matrix(one(T)*I, size(H.Q)...)

    H.x̄ .= alloc_x(x)
    H.x .= alloc_x(x)
    H.δ .= alloc_x(x)
    H.ḡ .= alloc_g(x)
    H.g .= alloc_g(x)
    H.γ .= alloc_g(x)

    H.x .= x
    H.g .= gradient(x)

    H
end

function compute_outer_products!(H::HessianBFGS)
    outer!(H.δγ, H.δ, H.γ)
    outer!(H.δδ, H.δ, H.δ)
end

function update!(H::HessianBFGS{T}, x::AbstractVector{T}; gradient = GradientAutodiff{T}(H.F, length(x))) where {T}
    # copy previous data and compute new gradient
    H.ḡ .= H.g
    H.x̄ .= H.x
    H.x .= x
    gradient(H.g, x)

    # δ = x - x̄
    direction(H) .= H.x - H.x̄

    # γ = g - ḡ
    H.γ .= H.g - H.ḡ

    # δγ = δ ⋅ γ
    δγ = compute_δγ(H)

    # BFGS
    # Q = Q - ... + ...
    # H.Q .-= (H.δ * H.γ' * H.Q .+ H.Q * H.γ * H.δ') ./ δγ .-
    #         (1 + dot(H.γ, H.Q, H.γ) ./ δγ) .* (H.δ * H.δ') ./ δγ
    
    if !iszero(δγ)
        compute_outer_products!(H)
        mul!(H.T1, H.δγ, H.Q)
        mul!(H.T2, H.Q, H.δγ')
        γQγ = compute_γQγ(H)
        H.T3 .= (one(T) + γQγ ./ δγ) .* H.δδ
        H.Q .-= (H.T1 .+ H.T2 .- H.T3) ./ δγ
    end

    H
end

Base.inv(H::HessianBFGS) = H.Q

Base.:\(H::HessianBFGS, b) = inv(H) * b