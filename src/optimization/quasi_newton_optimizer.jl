
struct NewtonOptimizerCache{T, AT <: AbstractArray{T}}
    x̄::AT
    x::AT
    δ::AT
    g::AT

    function NewtonOptimizerCache(x::AT) where {T, AT <: AbstractArray{T}}
        new{T,AT}(zero(x), zero(x), zero(x), zero(x))
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
    return cache
end

function initialize!(cache::NewtonOptimizerCache, x::AbstractVector)
    cache.x̄ .= eltype(x)(NaN)
    cache.x .= x
    cache.δ .= eltype(x)(NaN)
    cache.g .= eltype(x)(NaN)
    return cache
end

"create univariate objective for linesearch algorithm"
function linesearch_objective(objective::MultivariateObjective, cache::NewtonOptimizerCache)
    function ls_f(α)
        cache.x .= cache.x̄ .+ α .* cache.δ
        value(objective, cache.x)
    end

    UnivariateObjective(ls_f, 1.)
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

function NewtonOptimizerState(x::VT, objective::MultivariateObjective; hessian = HessianAD, linesearch = Bisection) where {XT, VT <: AbstractVector{XT}}
    cache = NewtonOptimizerCache(x)
    hess = hessian(objective, x)
    ls = LinesearchState(linesearch, linesearch_objective(objective, cache), x)

    NewtonOptimizerState(objective, hess, ls, cache)
end

# function NewtonOptimizerState(x::AbstractVector, objective::Function; kwargs...)
#     NewtonOptimizerState(x, MultivariateObjective(); kwargs...)
# end

NewtonOptimizer(args...; kwargs...) = NewtonOptimizerState(args...; kwargs...)
BFGSOptimizer(args...; kwargs...) = NewtonOptimizerState(args...; hessian = HessianBFGS, kwargs...)
DFPOptimizer(args...; kwargs...) = NewtonOptimizerState(args...; hessian = HessianDFP, kwargs...)

OptimizerState(algorithm::Newton, objective, x, y; kwargs...) = NewtonOptimizerState(x, objective; kwargs...)
OptimizerState(algorithm::BFGS, objective, x, y; kwargs...) = NewtonOptimizerState(x, objective; hessian = HessianBFGS, kwargs...)
OptimizerState(algorithm::DFP, objective, x, y; kwargs...) = NewtonOptimizerState(x, objective; hessian = HessianDFP, kwargs...)

objective(newton::NewtonOptimizerState) = newton.objective
hessian(newton::NewtonOptimizerState) = newton.hessian
cache(newton::NewtonOptimizerState) = newton.cache
linesearch(newton::NewtonOptimizerState) = newton.linesearch


function initialize!(newton::NewtonOptimizerState, x::AbstractVector)
    initialize!(hessian(newton), x)
    initialize!(cache(newton), x)
end


# function solver_step!(opt::Optimizer{<:Newton})
function solver_step!(x, newton::NewtonOptimizerState)
    # shortcuts
    x̄ = cache(newton).x̄
    δ = cache(newton).δ
    g = cache(newton).g

    # update cache
    update!(cache(newton), x, gradient!(objective(newton), x))

    # update Hessian
    update!(hessian(newton), x)

    # solve H δx = - ∇f
    ldiv!(δ, hessian(newton), g)
    δ .*= -1

    # apply line search
    α = newton.linesearch(1.0)
    x .= x̄ .+ α .* δ

    return x
end
