"""
    NewtonOptimizerState <: OptimizationAlgorithm

The optimizer state is needed to update the [`Optimizer`](@ref). This is different to [`OptimizerStatus`](@ref) and [`OptimizerResult`](@ref) which serve as diagnostic tools.
It stores a [`LinesearchState`](@ref) and a [`NewtonOptimizerCache`](@ref) which is used to compute the line search problem at each iteration.

# Keys

- `linesearch::`[`LinesearchState`](@ref)
- `cache::`[`NewtonOptimizerCache`](@ref)
"""
mutable struct NewtonOptimizerState{T, AT <: AbstractArray{T}, GT <: AbstractArray{T}} <: OptimizationAlgorithm
    x̄::AT
    ḡ::GT
    f̄::T

    function NewtonOptimizerState(x̄::AT, ḡ::GT, f̄::T) where {T, AT <: AbstractArray{T}, GT <: AbstractArray{T}}
        new{T, AT, GT}(x̄, ḡ, f̄)
    end
end

NewtonOptimizerState(x::AbstractVector{T}, g::AbstractVector{T}) where {T} = NewtonOptimizerState(copy(x), copy(g), zero(T))
NewtonOptimizerState(x::AbstractVector) = NewtonOptimizerState(copy(x), zero(x))

function initialize!(state::NewtonOptimizerState{T}, ::AbstractVector{T}) where {T}
    state.x̄ .= NaN
    state.ḡ .= NaN
    state.f̄ = NaN

    state
end

"""
    update!(state::NewtonOptimizerState, obj, x)

Update an instance of [`NewtonOptimizerState`](@ref) based on `x`, `g` and `hes`, where `g` can either be an `AbstractVector` or a [`Gradient`](@ref) and `hes` is a [`Hessian`](@ref).
This updates the [`NewtonOptimizerCache`](@ref) contained in the [`NewtonOptimizerState`](@ref) by calling [`update!(::NewtonOptimizerCache, ::AbstractVector, ::Union{AbstractVector, Gradient}, ::Hessian)`](@ref).

# Examples

We show that after initializing, update has to be called together with a [`Gradient`](@ref) and a [`Hessian`](@ref):

If we only call `update!` once there are still `NaN`s in the ...
```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: NewtonOptimizerState)
f(x) = sum(x.^2)
x = [1., 2.]
state = NewtonOptimizerState(x)
obj = OptimizerProblem(f, x)
grad = GradientAutodiff{Float64}(obj.F, length(x))
update!(state, obj, grad, x)

# output

NewtonOptimizerState{Float64, Vector{Float64}, Vector{Float64}}([1.0, 2.0], [2.0, 4.0], 0.0)
```
"""
function update!(state::NewtonOptimizerState, obj::OptimizerProblem, gradient::Gradient, x::AbstractVector)
    state.x̄ .= x
    state.ḡ .= gradient!(obj, gradient, x)

    state
end