"""
    NewtonOptimizerState <: OptimizationAlgorithm

The optimizer state is needed to update the [`Optimizer`](@ref). This is different to [`OptimizerStatus`](@ref) and [`OptimizerResult`](@ref) which serve as diagnostic tools.
It stores a [`LinesearchState`](@ref) and a [`NewtonOptimizerCache`](@ref) which is used to compute the line search problem at each iteration.

# Keys

- `linesearch::`[`LinesearchState`](@ref)
- `cache::`[`NewtonOptimizerCache`](@ref)
"""
struct NewtonOptimizerState{T, LS <: LinesearchState, NOC <: NewtonOptimizerCache{T}} <: OptimizationAlgorithm
    linesearch::LS
    cache::NOC

    function NewtonOptimizerState(linesearch::LS, cache::NOC) where {T, LS <: LinesearchState{T}, NOC <: NewtonOptimizerCache{T}}
        new{T,LS,NOC}(linesearch, cache)
    end
end

function NewtonOptimizerState(x::VT; linesearch::LinesearchMethod = Backtracking()) where {XT, VT <: AbstractVector{XT}}
    cache = NewtonOptimizerCache(x)
    initialize!(cache, x)
    ls = LinesearchState(linesearch; T = XT)

    NewtonOptimizerState(ls, cache)
end

cache(newton::NewtonOptimizerState) = newton.cache
direction(newton::NewtonOptimizerState) = direction(cache(newton))
gradient(newton::NewtonOptimizerState) = gradient(newton.cache)
hessian(newton::NewtonOptimizerState) = newton.hessian
linesearch(newton::NewtonOptimizerState) = newton.linesearch
rhs(newton::NewtonOptimizerState) = rhs(newton.cache)

function initialize!(newton::NewtonOptimizerState, x::AbstractVector)
    initialize!(cache(newton), x)

    newton
end

"""
    update!(state::NewtonOptimizerState, x, g, hes)

Update an instance of [`NewtonOptimizerState`](@ref) based on `x`, `g` and `hes`, where `g` can either be an `AbstractVector` or a [`Gradient`](@ref) and `hes` is a [`Hessian`](@ref).
This updates the [`NewtonOptimizerCache`](@ref) contained in the [`NewtonOptimizerState`](@ref) by calling [`update!(::NewtonOptimizerCache, ::AbstractVector, ::Union{AbstractVector, Gradient}, ::Hessian)`](@ref).

!!! info
    An instance of `NewtonOptimizerState` stores the `NewtonOptimizerCache` as well as a `LinesearchState`. The `LinesearchState` stays the same at every iteration, which is why only the `NewtonOptimizerState` is updated.

# Examples

We show that after initializing, update has to be called together with a [`Gradient`](@ref) and a [`Hessian`](@ref):

If we only call `update!` once there are still `NaN`s in the ...
```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: NewtonOptimizerState)
f(x) = sum(x.^2)
x = [1., 2.]
state = NewtonOptimizerState(x)
obj = OptimizerProblem(f, x)
g = gradient!(obj, x)
hes = HessianAutodiff(obj, x)
update!(hes, x)
update!(state, x, g, hes)

# output

NewtonOptimizerState{Float64, SimpleSolvers.BacktrackingState{Float64}, SimpleSolvers.NewtonOptimizerCache{Float64, Vector{Float64}}}(Backtracking with α₀ = 1.0, ϵ = 0.0001and p = 0.5., SimpleSolvers.NewtonOptimizerCache{Float64, Vector{Float64}}([1.0, 2.0], [1.0, 2.0], [-1.0, -2.0], [2.0, 4.0], [-2.0, -4.0]))
```
"""
function update!(state::NewtonOptimizerState, x::AbstractVector, g::Union{AbstractVector, Gradient}, hes::Hessian)
    update!(cache(state), x, g, hes)

    state
end