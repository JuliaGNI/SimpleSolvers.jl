
struct NewtonOptimizerCache{T, AT <: AbstractArray{T}}
    x̄::AT
    x::AT
    δ::AT
    g::AT
    rhs::AT

    function NewtonOptimizerCache(x::AT) where {T, AT <: AbstractArray{T}}
        new{T,AT}(zero(x), zero(x), zero(x), zero(x), zero(x))
    end
end

function update!(cache::NewtonOptimizerCache, x::AbstractVector)
    cache.x̄ .= x
    cache.x .= x
    cache.δ .= 0
    return cache
end

function update!(cache::NewtonOptimizerCache, x::AbstractVector, g::AbstractVector)
    update!(cache, x)
    cache.g .= g
    cache.rhs .= -g
    return cache
end

function initialize!(cache::NewtonOptimizerCache, x::AbstractVector)
    cache.x̄ .= eltype(x)(NaN)
    cache.x .= x
    cache.δ .= eltype(x)(NaN)
    cache.g .= eltype(x)(NaN)
    cache.rhs .= eltype(x)(NaN)
    return cache
end

"create univariate objective for linesearch algorithm"
function linesearch_objective(objective::MultivariateObjective, cache::NewtonOptimizerCache{T}) where {T}
    function f(α)
        cache.x .= cache.x̄ .+ α .* cache.δ
        value(objective, cache.x)
    end

    function d(α)
        cache.x .= cache.x̄ .+ α .* cache.δ
        dot(gradient!(objective, cache.x), cache.δ)
    end

    UnivariateObjective(f, d, one(T))
end


struct NewtonOptimizerState{OBJ <: MultivariateObjective, HES <: Hessian, LS <: LinesearchState, NOC <: NewtonOptimizerCache} <: OptimizationAlgorithm
    objective::OBJ
    hessian::HES
    linesearch::LS
    cache::NOC

    function NewtonOptimizerState(objective::OBJ, hessian::HES, linesearch::LS, cache::NOC) where {OBJ, HES, LS, NOC}
        new{OBJ,HES,LS,NOC}(objective, hessian, linesearch, cache)
    end
end

function NewtonOptimizerState(x::VT, objective::MultivariateObjective; hessian = HessianAD, linesearch = Backtracking()) where {XT, VT <: AbstractVector{XT}}
    cache = NewtonOptimizerCache(x)
    hess = hessian(objective, x)
    ls = LinesearchState(linesearch, linesearch_objective(objective, cache))

    NewtonOptimizerState(objective, hess, ls, cache)
end

NewtonOptimizer(args...; kwargs...) = NewtonOptimizerState(args...; kwargs...)
BFGSOptimizer(args...; kwargs...) = NewtonOptimizerState(args...; hessian = HessianBFGS, kwargs...)
DFPOptimizer(args...; kwargs...) = NewtonOptimizerState(args...; hessian = HessianDFP, kwargs...)

OptimizerState(algorithm::Newton, objective, x, y; kwargs...) = NewtonOptimizerState(x, objective; kwargs...)
OptimizerState(algorithm::BFGS, objective, x, y; kwargs...) = NewtonOptimizerState(x, objective; hessian = HessianBFGS, kwargs...)
OptimizerState(algorithm::DFP, objective, x, y; kwargs...) = NewtonOptimizerState(x, objective; hessian = HessianDFP, kwargs...)

cache(newton::NewtonOptimizerState) = newton.cache
direction(newton::NewtonOptimizerState) = cache(newton).δ
gradient(newton::NewtonOptimizerState) = newton.cache.g
hessian(newton::NewtonOptimizerState) = newton.hessian
linesearch(newton::NewtonOptimizerState) = newton.linesearch
objective(newton::NewtonOptimizerState) = newton.objective
rhs(newton::NewtonOptimizerState) = newton.cache.rhs


function initialize!(newton::NewtonOptimizerState, x::AbstractVector)
    initialize!(cache(newton), x)
    initialize!(hessian(newton), x)
end

function update!(newton::NewtonOptimizerState, x::AbstractVector)
    update!(cache(newton), x, gradient!(objective(newton), x))
    update!(hessian(newton), x)
end

function solver_step!(x::AbstractVector, newton::NewtonOptimizerState)
    # shortcut for Newton direction
    δ = direction(newton)

    # update cache and Hessian
    update!(newton, x)

    # solve H δx = - ∇f
    ldiv!(δ, hessian(newton), rhs(newton))

    # apply line search
    α = newton.linesearch()

    # compute new minimizer
    x .= x .+ α .* δ
end
