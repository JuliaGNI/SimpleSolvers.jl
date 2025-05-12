@doc raw"""
    linesearch_objective(objective, cache)

Create [`TemporaryUnivariateObjective`](@ref) for linesearch algorithm. The variable on which this objective depends is ``\alpha``.

# Example

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: NewtonOptimizerCache, linesearch_objective, update!)
x = [1, 0., 0.]
f = x -> sum(x .^ 3 / 6 + x .^ 2 / 2)
obj = MultivariateObjective(f, x)
gradient!(obj, x)
value!(obj, x)
cache = NewtonOptimizerCache(x)
hess = Hessian(obj, x; mode = :autodiff)
update!(hess, x)
update!(cache, x, obj.g, hess)
x₂ = [.9, 0., 0.]
gradient!(obj, x₂)
value!(obj, x₂)
update!(hess, x₂)
update!(cache, x₂, obj.g, hess)
ls_obj = linesearch_objective(obj, cache)
α = .1
(ls_obj.F(α), ls_obj.D(α))

# output

(0.4412947468016475, -0.8083161485821551)
```

In the example above we have to apply [`update!`](@ref) twice on the instance of [`NewtonOptimizerCache`](@ref) because it needs to store the current *and* the previous iterate.

# Implementation

Calling the function and derivative stored in the [`TemporaryUnivariateObjective`](@ref) created with `linesearch_objective` does not allocate a new array, but uses the one stored in the instance of [`NewtonOptimizerCache`](@ref).
"""
function linesearch_objective(objective::MultivariateObjective{T}, cache::NewtonOptimizerCache{T}) where {T}
    function f(α)
        cache.x .= compute_new_iterate(cache.x̄, α, direction(cache))
        value!(objective, cache.x)
    end

    function d(α)
        cache.x .= compute_new_iterate(cache.x̄, α, direction(cache))
        gradient!(objective, cache.x)
        cache.g .= objective.g
        dot(cache.g, direction(cache))
    end

    TemporaryUnivariateObjective{T}(f, d)
end

"""
    NewtonOptimizerState <: OptimizationAlgorithm

# Keys

- `objective::`[`MultivariateObjective`](@ref)
- `hessian::`[`Hessian`](@ref)
- `linesearch::`[`LinesearchState`](@ref)
- `ls_objective`
- `cache::`[`NewtonOptimizerCache`](@ref)
"""
struct NewtonOptimizerState{OBJ <: MultivariateObjective, HES <: Hessian, LS <: LinesearchState, LSO, NOC <: NewtonOptimizerCache} <: OptimizationAlgorithm
    objective::OBJ
    hessian::HES
    linesearch::LS
    ls_objective::LSO
    cache::NOC

    function NewtonOptimizerState(objective::OBJ, hessian::HES, linesearch::LS, ls_objetive::LSO, cache::NOC) where {OBJ, HES, LS, LSO, NOC}
        new{OBJ,HES,LS,LSO,NOC}(objective, hessian, linesearch, ls_objetive, cache)
    end
end

function NewtonOptimizerState(x::VT, objective::MultivariateObjective; mode = :autodiff, linesearch = Backtracking(), hessian = Hessian(objective, x; mode = mode)) where {XT, VT <: AbstractVector{XT}}
    cache = NewtonOptimizerCache(x, objective)
    initialize!(hessian, x)
    initialize!(cache, x)
    ls = LinesearchState(linesearch; T = XT)
    lso = linesearch_objective(objective, cache)

    NewtonOptimizerState(objective, hessian, ls, lso, cache)
end

# NewtonOptimizer(args...; kwargs...) = NewtonOptimizerState(args...; kwargs...)
# BFGSOptimizer(args...; kwargs...) = NewtonOptimizerState(args...; hessian = HessianBFGS, kwargs...)
# DFPOptimizer(args...; kwargs...) = NewtonOptimizerState(args...; hessian = HessianDFP, kwargs...)

OptimizationAlgorithm(algorithm::NMT, objective::AbstractObjective, x, y; kwargs...) where {NMT <: NewtonMethod} = error("OptimizationAlgorithm has to be extended to algorithm $(NMT).")
OptimizationAlgorithm(algorithm::NewtonMethod, objective::AbstractObjective, x; kwargs...) = OptimizationAlgorithm(algorithm, objective, x, objective(x); kwargs...)
OptimizationAlgorithm(algorithm::Newton, objective::AbstractObjective, x, y; kwargs...) = NewtonOptimizerState(x, objective; kwargs...)
OptimizationAlgorithm(algorithm::BFGS, objective::AbstractObjective, x, y; kwargs...) = NewtonOptimizerState(x, objective; hessian = HessianBFGS(objective, x), kwargs...)
OptimizationAlgorithm(algorithm::DFP, objective::AbstractObjective, x, y; kwargs...) = NewtonOptimizerState(x, objective; hessian = HessianDFP(objective, x), kwargs...)

cache(newton::NewtonOptimizerState) = newton.cache
direction(newton::NewtonOptimizerState) = cache(newton).δ
gradient(newton::NewtonOptimizerState) = gradient(newton.cache)
hessian(newton::NewtonOptimizerState) = newton.hessian
linesearch(newton::NewtonOptimizerState) = newton.linesearch
objective(newton::NewtonOptimizerState) = newton.objective
rhs(newton::NewtonOptimizerState) = rhs(newton.cache)

function initialize!(newton::NewtonOptimizerState, x::AbstractVector)
    initialize!(hessian(newton), x)
    initialize!(cache(newton), x)
end

"""
    update!(newton::NewtonOptimizerState, x)

Update an instance of [`NewtonOptimizerState`](@ref) based on `x`.
"""
function update!(newton::NewtonOptimizerState, x::AbstractVector)
    obj = objective(newton)
    update!(hessian(newton), x)
    update!(cache(newton), x, gradient!(obj, x), hessian(newton))
end

"""
    solver_step!(x, newton)

Compute a full iterate for an instance of [`NewtonOptimizerState`](@ref) `newton`.

This also performs a line search.
"""
function solver_step!(x::VT, newton::NewtonOptimizerState)::VT where {VT <: AbstractVector}
    # update cache and Hessian
    update!(newton, x)

    # solve H δx = - ∇f
    # rhs is -g
    ldiv!(direction(newton), hessian(newton), rhs(newton))

    # apply line search
    α = linesearch(newton)(newton.ls_objective)

    # compute new minimizer
    x .= compute_new_iterate(x, α, direction(newton))
end
