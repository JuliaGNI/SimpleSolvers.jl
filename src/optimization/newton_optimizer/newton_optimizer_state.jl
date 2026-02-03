"""
    NewtonOptimizerState <: OptimizerState

The optimizer state is needed to update the [`Optimizer`](@ref). This is different to [`OptimizerStatus`](@ref) and [`OptimizerResult`](@ref) which serve as diagnostic tools.

Note that this is also used for the [`BFGS`](@ref) and the [`DFP`](@ref) optimizer.

# Keys

- `x̄`
- `ḡ`
- `f̄`
"""
mutable struct NewtonOptimizerState{T,AT<:AbstractArray{T},GT<:AbstractArray{T}} <: OptimizerState{T}
    iterations::Int

    x::AT
    x̄::AT
    g::GT
    ḡ::GT
    f::T
    f̄::T

    function NewtonOptimizerState(X::AT, G::GT) where {T,AT<:AbstractArray{T},GT<:AbstractArray{T}}
        x = zero(X)
        x̄ = zero(X)
        g = zero(X)
        ḡ = zero(X)
        x .= T(NaN)
        x̄ .= T(NaN)
        g .= T(NaN)
        ḡ .= T(NaN)
        new{T,AT,GT}(0, x, x̄, g, ḡ, T(NaN), T(NaN))
    end

    NewtonOptimizerState(x) = NewtonOptimizerState(x, x)
end

OptimizerState(::Newton, x_args...) = NewtonOptimizerState(x_args...)

function initialize!(state::NewtonOptimizerState{T}, x::AbstractVector{T}, g::AbstractVector{T}, f::T) where {T}
    state.iterations = 0
    state.x .= x
    state.g .= g
    state.f = f
    state.x̄ .= T(NaN)
    state.ḡ .= T(NaN)
    state.f̄ = T(NaN)
end

function update!(state::NewtonOptimizerState{T}, x::AbstractVector{T}, g::AbstractVector{T}, f::T) where {T}
    state.x̄ .= state.x
    state.ḡ .= state.g
    state.f̄ = state.f
    state.x .= x
    state.g .= g
    state.f = f
end

"""
    update!(state::NewtonOptimizerState, gradient, x)

Update an instance of [`NewtonOptimizerState`](@ref) based on `x`, `g` and `hes`, where `g` can either be an `AbstractVector` or a [`Gradient`](@ref) and `hes` is a [`Hessian`](@ref).
This updates the [`NewtonOptimizerCache`](@ref) contained in the [`NewtonOptimizerState`](@ref) by calling [`update!(::NewtonOptimizerCache, ::AbstractVector, ::Union{AbstractVector, Gradient}, ::Hessian)`](@ref).

# Examples

We show that after initializing, update has to be called together with a [`Gradient`](@ref) and a [`Hessian`](@ref):

If we only call `update!` once there are still `NaN`s in the ...
```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: NewtonOptimizerState)
f(x) = sum(x.^2)
x = [1., 2.]
state = NewtonOptimizerState(x)
grad = GradientAutodiff{Float64}(f, length(x))
update!(state, grad, x)

# output

NewtonOptimizerState{Float64, Vector{Float64}, Vector{Float64}}(0, [1.0, 2.0], [NaN, NaN], [2.0, 4.0], [NaN, NaN], 5.0, NaN)
```
"""
function update!(state::NewtonOptimizerState, gradient::Gradient, x::AbstractVector)
    update!(state, x, gradient(x), gradient.F(x))

    state
end
