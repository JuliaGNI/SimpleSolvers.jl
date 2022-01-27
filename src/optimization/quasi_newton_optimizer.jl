
mutable struct QuasiNewtonOptimizer{XT, YT, OT <: MultivariateObjective, HT <: HessianParameters, TS <: LineSearch, AT <: AbstractVector{XT}} <: Optimizer{XT}
    objective::OT
    hessian::HT
    linesearch::TS
    cache::NewtonOptimizerCache{XT,AT}
    config::Options{XT}
    status::OptimizerStatus{XT,YT,AT}

    function QuasiNewtonOptimizer{XT,YT,OT,HT,TS,AT}(objective, hessian, linesearch, cache, config, status) where {XT,YT,OT,HT,TS,AT}
        new(objective, hessian, linesearch, cache, config, status)
    end
end

function QuasiNewtonOptimizer(x::VT, F; ∇F! = nothing, hessian = HessianParametersAD, linesearch = Bisection, config = Options()) where {XT, VT <: AbstractVector{XT}}
    G = GradientParameters(∇F!, F, x)

    objective = MultivariateObjective(F, G, x)
    hessian = hessian(F, x)

    YT = typeof(F(x))
    cache = NewtonOptimizerCache(x)
    status = OptimizerStatus{XT,YT,VT}(config, length(x))

    # create objective for linesearch algorithm
    function ls_f(α)
        cache.x .= cache.x̄ .+ α .* cache.δ
        value(objective, cache.x)
    end

    ls_objective = UnivariateObjective(ls_f, 1.)

    # create linesearch algorithm
    ls = linesearch(ls_objective)

    QuasiNewtonOptimizer{XT, YT, typeof(objective), typeof(hessian), typeof(ls), VT}(objective, hessian, ls, cache, config, status)
end

BFGSOptimizer(args...; kwargs...) = QuasiNewtonOptimizer(args...; hessian = HessianBFGS, kwargs...)
DFPOptimizer(args...; kwargs...) = QuasiNewtonOptimizer(args...; hessian = HessianDFP, kwargs...)

cache(s::QuasiNewtonOptimizer) = s.cache
config(s::QuasiNewtonOptimizer) = s.config
status(s::QuasiNewtonOptimizer) = s.status
objective(s::QuasiNewtonOptimizer) = s.objective
hessian(s::QuasiNewtonOptimizer) = s.hessian

check_gradient(s::QuasiNewtonOptimizer) = check_gradient(gradient(objective(s)))
print_gradient(s::QuasiNewtonOptimizer) = print_gradient(gradient(objective(s)))
print_status(s::QuasiNewtonOptimizer) = print_status(status(s), config(s))

assess_convergence(s::QuasiNewtonOptimizer) = assess_convergence(status(s), config(s))

function initialize!(s::QuasiNewtonOptimizer{T}, x₀::Vector{T}) where {T}
    clear!(objective(s))
    value!(objective(s), x₀)
    gradient!(objective(s), x₀)
    initialize!(hessian(s), x₀)
    initialize!(status(s), x₀, value(objective(s)), gradient(objective(s)))
end


function solver_step!(s::QuasiNewtonOptimizer{T}) where {T}
    # update Hessian
    update!(hessian(s), status(s))

    # update cache
    update!(cache(s), status(s))

    # solve H δx = - ∇f
    ldiv!(s.cache.δ, hessian(s), s.status.g)
    s.cache.δ .*= -1

    # apply line search
    α, f = s.linesearch(1.0)
    # s.cache.x .= s.cache.x̄ .+ α .* s.cache.δ

    # compute gradient at new solution and update residual
    residual!(status(s), s.cache.x, f, gradient!(objective(s), s.cache.x))
end
