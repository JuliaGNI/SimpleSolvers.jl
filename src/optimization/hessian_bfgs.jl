"""
    HessianBFGS <: Hessian

A `struct` derived from [`Hessian`](@ref) to be used for an [`Optimizer`](@ref).

# Fields
- `problem::`[`MultivariateOptimizerProblem`](@ref): 
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
struct HessianBFGS{T,VT,MT,OBJ} <: IterativeHessian{T}
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

    function HessianBFGS(problem::MultivariateOptimizerProblem, x::AbstractVector{T}) where {T}
        Q  = alloc_h(x)
        
        T1 = zero(Q)
        T2 = zero(Q)
        T3 = zero(Q)
        δγ = zero(Q)
        δδ = zero(Q)
            
        new{T,typeof(x),typeof(Q),typeof(problem)}(problem, zero(x), zero(x), zero(x), zero(x), zero(x), zero(x), Q, T1, T2, T3, δγ, δδ)
    end
end

HessianBFGS(F::Callable, x::AbstractVector) = HessianBFGS(MultivariateOptimizerProblem(F, x), x)

Hessian(::BFGS, ForOBJ::Union{Callable, MultivariateOptimizerProblem}, x::AbstractVector) = HessianBFGS(ForOBJ, x)

@doc raw"""
    initialize!(H, x)

Initialize an object `H` of type [`HessianBFGS`](@ref). 

We note that unlike most other [`initialize!`](@ref) methods this one is not writing `NaN`s everywhere - ``Q = H^{-1}`` is set to the identity.

# Examples

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: initialize!)
f(x) = x .^ 2
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
function initialize!(H::HessianBFGS, x::AbstractVector)
    H.Q .= Matrix(1.0I, size(H.Q)...)

    H.x̄ .= alloc_x(x)
    H.x .= alloc_x(x)
    H.δ .= alloc_x(x)
    H.ḡ .= alloc_g(x)
    H.g .= alloc_g(x)
    H.γ .= alloc_g(x)

    H
end

function compute_outer_products!(H::HessianBFGS)
    outer!(H.δδ, H.δ, H.δ)
    outer!(H.δγ, H.δ, H.γ)
end

function update!(H::HessianBFGS, x::AbstractVector)
    update!(H, x, gradient!(problem(H), x))

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
        H.T3 .= (1 + γQγ ./ δγ) .* H.δδ
        H.Q .-= (H.T1 .+ H.T2 .- H.T3) ./ δγ
    end

    H
end

Base.inv(H::HessianBFGS) = H.Q

Base.:\(H::HessianBFGS, b) = inv(H) * b

LinearAlgebra.ldiv!(x, H::HessianBFGS, b) = mul!(x, inv(H), b)
