
mutable struct QuasiNewtonOptimizer{XT, YT, FT <: Callable, GT <: GradientParameters, HT <: HessianParameters, TS <: LineSearch, VT <: AbstractVector{XT}} <: Optimizer{XT}
    x̃::VT
    ỹ::YT
    g̃::VT

    F::FT
    G::GT
    H::HT

    ls::TS

    params::NonlinearSolverParameters{XT}
    status::OptimizerStatus{XT,YT,VT}

    function QuasiNewtonOptimizer{XT,YT,FT,GT,HT,TS,VT}(x, F, G, H, line_search) where {XT,YT,FT,GT,HT,TS,VT}
        x̃ = zero(x)
        g̃ = zero(x)

        params = NonlinearSolverParameters(XT)
        status = OptimizerStatus{XT,YT,VT}(length(x))
    
        new(x̃, zero(YT), g̃, F, G, H, line_search, params, status)
    end
end

function QuasiNewtonOptimizer(x::VT, F::Function; ∇F!::Union{Callable,Nothing} = nothing, hessian = HessianBFGS, linesearch = Bisection(F)) where {XT, VT <: AbstractVector{XT}}
    G = getGradientParameters(∇F!, F, x)
    H = hessian(x)
    YT = typeof(F(x))
    QuasiNewtonOptimizer{XT,YT,typeof(F),typeof(G),typeof(H),typeof(linesearch),VT}(x, F, G, H, linesearch)
end

BFGSOptimizer(args...; kwargs...) = QuasiNewtonOptimizer(args...; hessian = HessianBFGS, kwargs...)
DFPOptimizer(args...; kwargs...) = QuasiNewtonOptimizer(args...; hessian = HessianDFP, kwargs...)


status(s::QuasiNewtonOptimizer) = s.status
params(s::QuasiNewtonOptimizer) = s.params
gradient(s::QuasiNewtonOptimizer) = s.G
hessian(s::QuasiNewtonOptimizer) = s.H

check_gradient(s::QuasiNewtonOptimizer) = check_gradient(s.g)
print_gradient(s::QuasiNewtonOptimizer) = print_gradient(s.g)

print_solver_status(s::QuasiNewtonOptimizer) = print_solver_status(status(s), params(s))
check_solver_converged(s::QuasiNewtonOptimizer) = check_solver_converged(status(s), params(s))


function setInitialConditions!(s::QuasiNewtonOptimizer{T}, x₀::Vector{T}) where {T}
    s.x̃ .= x₀
    s.ỹ  = s.F(s.x̃)
    computeGradient(s.x̃, s.g̃, s.G)
    initialize!(status(s), s.x̃, s.ỹ, s.g̃)
    initialize!(hessian(s))
end


function _f(s, α)
    s.x̃ .= s.status.x .+ α .* s.status.δ
    s.F(s.x̃)
end

function _linesearch!(s::QuasiNewtonOptimizer)
    mul!(s.status.δ, inverse(s.H), s.status.g)
    s.status.δ .*= -1
    objective = α -> _f(s, α)
    a, b = bracket_minimum(objective, 1.0)
    α, s.status.y = s.ls(objective, a, b)
    s.status.x .= s.status.x̄ .+ α .* s.status.δ
end


function solve!(s::QuasiNewtonOptimizer{T}; n::Int = 0) where {T}
    local nmax::Int = n > 0 ? nmax = n : s.params.nmax

    s.status.i = 0
    if s.status.rgₐ ≥ s.params.atol² || n > 0 || s.params.nmin > 0
        for s.status.i = 1:nmax
            # update status and temporaries
            update!(status(s))

            # apply line search
            _linesearch!(s)

            # compute Gradient at new solution
            computeGradient(s.status.x, s.status.g, s.G)

            # compute residual
            residual!(s.status)

            # println(s.status.i, ", ", s.x̄, ", ", s.ȳ, ", ", s.status.rₐ,", ",  s.status.rᵣ,", ",  s.status.rₛ)
            if check_solver_converged(s.status, s.params) && s.status.i ≥ s.params.nmin && !(n > 0)
                if s.params.nwarn > 0 && s.status.i ≥ s.params.nwarn
                    println("WARNING: BFGS Optimizer took ", s.status.i, " iterations.")
                end
                break
            end

            # update Hessian
            update!(s.H, s.status)
        end
    end
end

function solve!(x, s::QuasiNewtonOptimizer; kwargs...)
    setInitialConditions!(s, x)
    solve!(s; kwargs...)
    x .= s.params.x
end
