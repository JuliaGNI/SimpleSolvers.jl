"""
    BFGSState <: OptimizerState

The [`OptimizerState`](@ref) corresponding to the [`BFGS`](@ref) method.

# Keys
- `x̄`
- `ḡ`
- `f̄`
- `Q`
"""
mutable struct BFGSState{T, AT <: AbstractArray{T}, GT <: AbstractArray{T}, MT <: AbstractMatrix{T}} <: OptimizerState{T}
    x̄::AT
    ḡ::GT
    f̄::T
    Q::MT

    function BFGSState(x̄::AT, ḡ::GT, f̄::T, Q::MT) where {T, AT <: AbstractArray{T}, GT <: AbstractArray{T}, MT <: AbstractMatrix{T}}
        state = new{T, AT, GT, MT}(x̄, ḡ, f̄, Q)
        initialize!(state, x̄)
        state
    end
end

BFGSState(x̄::AbstractVector{T}, ḡ::AbstractVector{T}, f̄::T) where {T} = BFGSState(copy(x̄), copy(ḡ), f̄, alloc_h(x̄))
BFGSState(x̄::AbstractVector{T}, ḡ::AbstractVector{T}) where {T} = BFGSState(copy(x̄), copy(ḡ), zero(T))
BFGSState(x̄::AbstractVector) = BFGSState(copy(x̄), copy(x̄))

OptimizerState(::BFGS, x_args...) = BFGSState(x_args...)

inverse_hessian(state::BFGSState) = state.Q

function initialize!(state::BFGSState{T}, ::AbstractVector{T}) where {T}
    state.x̄ .= NaN
    state.ḡ .= NaN
    state.f̄ = NaN
    inverse_hessian(state) .= one(inverse_hessian(state))

    state
end

function update!(state::BFGSState, gradient::Gradient, x::AbstractVector)
    state.x̄ .= x
    gradient(state.ḡ, x)
    state.f̄ = gradient.F(x)

    state
end