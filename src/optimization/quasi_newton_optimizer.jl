
mutable struct QuasiNewtonOptimizer{XT, YT, OT <: MultivariateObjective, TS <: LineSearch, VT <: AbstractVector{XT}} <: Optimizer{XT}
    x̃::VT
    g̃::VT

    objective::OT

    ls::TS

    params::NonlinearSolverParameters{XT}
    status::OptimizerStatus{XT,YT,VT}

    function QuasiNewtonOptimizer{XT,YT,OT,TS,VT}(x, objective, status, line_search) where {XT,YT,OT,TS,VT}
        x̃ = zero(x)
        g̃ = zero(x)

        params = NonlinearSolverParameters(XT)
    
        new(x̃, g̃, objective, line_search, params, status)
    end
end

function QuasiNewtonOptimizer(x::VT, F::Callable; ∇F!::Union{Callable,Nothing} = nothing, hessian = HessianParametersAD, linesearch = Bisection) where {XT, VT <: AbstractVector{XT}}
    G = GradientParameters(∇F!, F, x)
    H = hessian(F, x)

    objective = MultivariateObjective(F, G, H, x)

    YT = typeof(F(x))
    status = OptimizerStatus{XT,YT,VT}(length(x))

    # create objective for linesearch algorithm
    ls_x̃ = zero(x)

    function ls_f(α)
        ls_x̃ .= status.x .+ α .* status.δ
        value(objective, F(ls_x̃))
    end

    ls_objective = UnivariateObjective(ls_f, 1.)

    # create linesearch algorithm
    ls = linesearch(ls_objective)

    QuasiNewtonOptimizer{XT,YT,typeof(objective),typeof(ls),VT}(x, objective, status, ls)
end

BFGSOptimizer(args...; kwargs...) = QuasiNewtonOptimizer(args...; hessian = HessianBFGS, kwargs...)
DFPOptimizer(args...; kwargs...) = QuasiNewtonOptimizer(args...; hessian = HessianDFP, kwargs...)


status(s::QuasiNewtonOptimizer) = s.status
params(s::QuasiNewtonOptimizer) = s.params
objective(s::QuasiNewtonOptimizer) = s.objective
# gradient(s::QuasiNewtonOptimizer) = (g,x) -> g .= gradient(x, objective(s))
# hessian(s::QuasiNewtonOptimizer) = (h,x) -> h .= hessian(x, objective(s))

check_gradient(s::QuasiNewtonOptimizer) = check_gradient(s.g)
print_gradient(s::QuasiNewtonOptimizer) = print_gradient(s.g)

print_solver_status(s::QuasiNewtonOptimizer) = print_solver_status(status(s), params(s))
check_solver_converged(s::QuasiNewtonOptimizer) = check_solver_converged(status(s), params(s))


function initialize!(s::QuasiNewtonOptimizer{T}, x₀::Vector{T}) where {T}
    clear!(objective(s))
    value!(objective(s), x₀)
    gradient!(objective(s), x₀)
    initialize!(status(s), x₀, value(objective(s)), gradient(objective(s)))
    initialize!(objective(s).H, x₀)
end


function solve!(s::QuasiNewtonOptimizer{T}; n::Int = 0) where {T}
    local nmax::Int = n > 0 ? nmax = n : s.params.nmax

    s.status.i = 0
    if s.status.rgₐ ≥ s.params.atol² || n > 0 || s.params.nmin > 0
        for s.status.i = 1:nmax
            # update status and temporaries
            update!(status(s))

            # solve H δx = - ∇f
            ldiv!(s.status.δ, s.objective.H, s.status.g)
            s.status.δ .*= -1
        
            # apply line search
            α, y = s.ls(1.0)
            s.status.x .= s.status.x̄ .+ α .* s.status.δ
            s.status.y  = y
        
            # compute Gradient at new solution
            s.status.g .= gradient!(s.objective, s.status.x)

            # compute residual
            residual!(s.status)

            # check for convergence
            if check_solver_converged(s.status, s.params) && s.status.i ≥ s.params.nmin && !(n > 0)
                if s.params.nwarn > 0 && s.status.i ≥ s.params.nwarn
                    println("WARNING: Quasi-Newton Optimizer took ", s.status.i, " iterations.")
                end
                break
            end

            # update Hessian
            update!(s.objective.H, status(s))
        end
    end
end

function solve!(x, s::QuasiNewtonOptimizer; kwargs...)
    initialize!(s, x)
    solve!(s; kwargs...)
    x .= s.params.x
end
