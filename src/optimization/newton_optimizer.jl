"""
    NewtonOptimizerCache

# Keys

- `x̄`: the previous iterate,
- `x`: current iterate,
- `δ`: direction of optimization step (difference between `x` and `x̄`); this is obtained by multiplying `rhs` with the inverse of the Hessian,
- `g`: gradient value,
- `rhs`: the right hand side used to compute the update.

To understand how these are used in practice see e.g. [`linesearch_objective`](@ref).
"""
struct NewtonOptimizerCache{T, AT <: AbstractArray{T}}
    x̄::AT
    x::AT
    δ::AT
    g::AT
    rhs::AT

    function NewtonOptimizerCache(x::AT) where {T, AT <: AbstractArray{T}}
        cache = new{T,AT}(similar(x), similar(x), similar(x), similar(x), similar(x))
        initialize!(cache, x)
        cache
    end

    # we probably don't need this constructor
    function NewtonOptimizerCache(x::AT, objective::MultivariateObjective) where {T <: Number, AT <: AbstractArray{T}}
        g = gradient!(objective, x)
        new{T, AT}(copy(x), copy(x), zero(x), g, -g)
    end
end

"""
    rhs(cache)

Return the right hand side of an instance of [`NewtonOptimizerCache`](@ref)
"""
rhs(cache::NewtonOptimizerCache) = cache.rhs
"""
    gradient(::NewtonOptimizerCache)

Return the stored gradient (array) of an instance of [`NewtonOptimizerCache`](@ref)
"""
gradient(cache::NewtonOptimizerCache) = cache.g
"""
    direction(::NewtonOptimizerCache)

Return the direction of the gradient step (i.e. `δ`) of an instance of [`NewtonOptimizerCache`](@ref).
"""
direction(cache::NewtonOptimizerCache) = cache.δ

function update!(cache::NewtonOptimizerCache, x::AbstractVector)
    cache.x̄ .= x
    cache.x .= x
    direction(cache) .= 0
    cache
end

function update!(cache::NewtonOptimizerCache, x::AbstractVector, g::AbstractVector)
    update!(cache, x)
    gradient(cache) .= g
    rhs(cache) .= -g
    cache
end

update!(cache::NewtonOptimizerCache, x::AbstractVector, g::Gradient) = update!(cache, x, gradient(x, g))

@doc raw"""
    update!(cache::NewtonOptimizerCache, x, g, hes)

Update an instance of [`NewtonOptimizerCache`](@ref) based on `x`.

This sets:
```math
\bar{x}^\mathtt{cache} \gets x,
x^\mathtt{cache} \gets x,
g^\mathtt{cache} \gets g,
\mathrm{rhs}^\mathtt{cache} \gets -g,
\delta^\mathtt{cache} \gets H^{-1}\mathrm{rhs}^\mathtt{cache},
```
where we wrote ``H`` for the Hessian (i.e. the input argument `hes`). 

Also see [`update!(::NewtonSolverCache, ::AbstractVector)`](@ref). 

# Implementation

The multiplication by the inverse of ``H`` is done with `LinearAlgebra.ldiv!`.
"""
function update!(cache::NewtonOptimizerCache, x::AbstractVector, g::Union{AbstractVector, Gradient}, hes::Hessian)
    update!(cache, x, g)
    ldiv!(direction(cache), hes, rhs(cache))
    cache
end

function initialize!(cache::NewtonOptimizerCache, x::AbstractVector)
    cache.x̄ .= alloc_x(x)
    cache.x .= copy(x)
    cache.δ .= alloc_x(x)
    cache.g .= alloc_g(x)
    cache.rhs .= alloc_g(x)
    cache
end

@doc raw"""
    linesearch_objective(objective, cache)

Create [`TemporaryUnivariateObjective`](@ref) for linesearch algorithm. The variable on which this objective depends is ``\alpha``.

# Example

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: NewtonOptimizerCache, linesearch_objective, update!)
x = [1., 0., 0.]
f = x -> sum(x .^ 3 / 6 + x .^ 2 / 2)
obj = MultivariateObjective(f, x)
gradient!(obj, x)
value!(obj, x)
cache = NewtonOptimizerCache(x)
update!(cache, x, obj.g)
x₂ = [.9, 0., 0.]
gradient!(obj, x₂)
value!(obj, x₂)
update!(cache, x₂, obj.g)
ls_obj = linesearch_objective(obj, cache)
α = .1
(ls_obj.F(α), ls_obj.D(α))

# output

(0.5265000000000001, -1.7030250000000005)
```

In the example above we have to apply [`update!`](@ref) twice on the instance of [`NewtonOptimizerCache`](@ref) because it needs to store the current *and* the previous iterate.

# Implementation

Calling the function and derivative stored in the [`TemporaryUnivariateObjective`](@ref) created with `linesearch_objective` does not allocate a new array, but uses the one stored in the instance of [`NewtonOptimizerCache`](@ref).
"""
function linesearch_objective(objective::MultivariateObjective, cache::NewtonOptimizerCache{T}) where {T}
    function f(α)
        cache.x .= compute_new_iterate(cache.x̄, α, direction(cache))
        objective.F(cache.x)
    end

    function d(α)
        cache.x .= compute_new_iterate(cache.x̄, α, direction(cache))
        gradient!(objective, cache.x)
        cache.g .= objective.g
        dot(gradient!(objective, cache.x), cache.rhs)
    end

    TemporaryUnivariateObjective(f, d)
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
    ls = LinesearchState(linesearch)
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
