"""
    NewtonOptimizerCache

# Keys

- `x̄`
- `x`: current iterate
- `δ`: direction of optimization step (difference between `x` and `x̄`); this is obtained by multiplying `rhs` with the inverse of the Hessian.
- `g`: gradient value
- `rhs`
"""
struct NewtonOptimizerCache{T, AT <: AbstractArray{T}}
    x̄::AT
    x::AT
    δ::AT
    g::AT
    rhs::AT

    function NewtonOptimizerCache(x::AT) where {T, AT <: AbstractArray{T}}
        new{T,AT}(zero(x), zero(x), zero(x), zero(x), zero(x))
    end

    function NewtonOptimizerCache(x::AT, objective::MultivariateObjective) where {T <: Number, AT <: AbstractArray{T}}
        g = gradient!(objective, x)
        new{T, AT}(x, x, zero(x), g, -g)
    end
end

rhs(cache::NewtonOptimizerCache) = cache.rhs
gradient(cache::NewtonOptimizerCache) = cache.g
direction(cache::NewtonOptimizerCache) = cache.δ

function update!(cache::NewtonOptimizerCache, x::AbstractVector)
    cache.x̄ .= x
    cache.x .= x
    cache.δ .= 0
    cache
end

function update!(cache::NewtonOptimizerCache, x::AbstractVector, g::AbstractVector)
    update!(cache, x)
    gradient(cache) .= g
    rhs(cache) .= -g
    cache
end

function initialize!(cache::NewtonOptimizerCache, x::AbstractVector)
    cache.x̄ .= eltype(x)(NaN)
    cache.x .= x
    cache.δ .= eltype(x)(NaN)
    cache.g .= eltype(x)(NaN)
    cache.rhs .= eltype(x)(NaN)
    cache
end

function initialize!(cache::NewtonOptimizerCache, x::AbstractVector, objective::MultivariateObjective, hes::Hessian)
    cache.x̄ .= x
    cache.x .= x
    gradient(cache) .= gradient!(objective, x)
    rhs(cache) .= -gradient(cache)
    ldiv!(cache.δ, hes, gradient(cache))
    cache
end

@doc raw"""
    linesearch_objective(objective!, jacobian!, cache)

Create [`TemporaryUnivariateObjective`](@ref) for linesearch algorithm. The variable on which this objective depends is ``\alpha``.

# Example

"""
function linesearch_objective(objective::MultivariateObjective, cache::NewtonOptimizerCache{T}) where {T}
    function f(α)
        x = cache.x̄ .+ α .* direction(cache)
        objective.F(x)
    end

    function d(α)
        x = cache.x̄ .+ α .* direction(cache)
        gradient!(objective, cache.x̄)
        g = objective.g
        dot(gradient!(objective, x), g)
    end

    UnivariateObjective(f, d, f(zero(T)), d(zero(T)), zero(T), zero(T), 0, 0)
end

"""
    NewtonOptimizerState <: OptimizationAlgorithm

# Keys

- `objective::`[`MultivariateObjective`](@ref)
- `hessian::`[`Hessian`](@ref)
- `linesearch::`[`LinesearchState`]
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

function NewtonOptimizerState(x::VT, objective::MultivariateObjective; mode = :autodiff, linesearch = Backtracking(), hesssian = Hessian(objective, x; mode = mode)) where {XT, VT <: AbstractVector{XT}}
    cache = NewtonOptimizerCache(x, objective)
    initialize!(hessian, x)
    initialize!(cache, x, objective, hessian)
    ls = LinesearchState(linesearch)
    lso = linesearch_objective(objective, cache)

    NewtonOptimizerState(objective, hessian, ls, lso, cache)
end

NewtonOptimizer(args...; kwargs...) = NewtonOptimizerState(args...; kwargs...)
BFGSOptimizer(args...; kwargs...) = NewtonOptimizerState(args...; hessian = HessianBFGS, kwargs...)
DFPOptimizer(args...; kwargs...) = NewtonOptimizerState(args...; hessian = HessianDFP, kwargs...)

OptimizationAlgorithm(algorithm::NMT, objective::AbstractObjective, x, y; kwargs...) where {NMT <: NewtonMethod} = error("OptimizationAlgorithm has to be extended to algorithm $(NMT).")
OptimizationAlgorithm(algorithm::NewtonMethod, objective::AbstractObjective, x; kwargs...) = OptimizationAlgorithm(algorithm, objective, x, objective(x); kwargs...)
OptimizationAlgorithm(algorithm::Newton, objective::AbstractObjective, x, y; kwargs...) = NewtonOptimizerState(x, objective; kwargs...)
OptimizationAlgorithm(algorithm::BFGS, objective::AbstractObjective, x, y; kwargs...) = NewtonOptimizerState(x, objective; hessian = HessianBFGS, kwargs...)
OptimizationAlgorithm(algorithm::DFP, objective::AbstractObjective, x, y; kwargs...) = NewtonOptimizerState(x, objective; hessian = HessianDFP, kwargs...)

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

function update!(newton::NewtonOptimizerState, x::AbstractVector)
    update!(cache(newton), x, gradient!(objective(newton), x))
    update!(hessian(newton), x)
end

function solver_step!(x::AbstractVector, newton::NewtonOptimizerState)
    # update cache and Hessian
    update!(newton, x)

    # solve H δx = - ∇f
    # rhs is -g
    ldiv!(direction(newton), hessian(newton), rhs(newton))

    # apply line search
    α = linesearch(newton)(newton.ls_objective)

    # compute new minimizer
    x .= x .+ α .* direction(newton)
end
