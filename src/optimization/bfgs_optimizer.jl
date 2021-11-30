mutable struct BFGSOptimizer{XT, YT, FT <: Callable, T∇ <: GradientParameters, TS <: LineSearch, VT <: AbstractVector{XT}} <: Optimizer{XT}
    x::VT
    x̄::VT
    δ::VT

    g::VT
    ḡ::VT
    γ::VT

    Q::Matrix{XT}
    y::YT
    ȳ::YT

    F::FT
    ∇params::T∇

    ls::TS

    params::NonlinearSolverParameters{XT}
    status::NonlinearSolverStatus{XT}

    function BFGSOptimizer{XT,YT,FT,T∇,TS,VT}(x, F, ∇params, line_search) where {XT,YT,FT,T∇,TS,VT}
        Q = zeros(XT, length(x), length(x))
    
        x = zero(x)
        x̄ = zero(x)
        δ = zero(x)
    
        g = zero(x)
        ḡ = zero(x)
        γ = zero(x)

        nls_params = NonlinearSolverParameters(XT)
        nls_status = NonlinearSolverStatus{XT}(length(x))
    
        new(x, x̄, δ, g, ḡ, γ, Q, zero(YT), zero(YT), F, ∇params, line_search, nls_params, nls_status)
    end
end


function BFGSOptimizer(x::VT, F::Function; ∇F!::Union{Callable,Nothing} = nothing, linesearch = Bisection(F)) where {XT, VT <: AbstractVector{XT}}
    ∇params = getGradientParameters(∇F!, F, x)
    YT = typeof(F(x))
    BFGSOptimizer{XT,YT,typeof(F),typeof(∇params),typeof(linesearch),VT}(x, F, ∇params, linesearch)
end


status(solver::BFGSOptimizer) = solver.status
params(solver::BFGSOptimizer) = solver.params

function computeGradient(s::BFGSOptimizer)
    computeGradient(s.x, s.g, s.∇params)
end

check_gradient(s::BFGSOptimizer) = check_gradient(s.g)
print_gradient(s::BFGSOptimizer) = print_gradient(s.g)


function setInitialConditions!(s::BFGSOptimizer{T}, x₀::Vector{T}) where {T}
    s.x .= x₀
    s.y = s.F(s.x)
    computeGradient(s)
    s.Q .= Matrix(1.0I, length(s.x), length(s.x))
end


function _residual_initial!(status, x, y, g)
    status.rₐ = Inf
    status.rᵣ = Inf
    status.rₛ = Inf
    status.x₀ .= x
    status.xₚ .= x
    status.y₀ .= y
    status.yₚ .= y
end


function _residual!(status, x̄, ȳ, ḡ)
    status.rₐ = norm(status.yₚ[1] - ȳ[1])
    status.rᵣ = norm(status.yₚ[1] - ȳ[1]) / norm(status.yₚ[1])
    status.rₛ = norm(status.xₚ .- x̄)
end


function _linesearch!(x̄, f, x, d)
    objective = α -> f(x .+ α .* d)
    a, b = bracket_minimum(objective, 1.0)
    α = bisection(objective, a, b)
    x̄ .= x .+ α .* d
end


function solve!(s::BFGSOptimizer{T}; n::Int = 0) where {T}
    local nmax::Int = n > 0 ? nmax = n : s.params.nmax

    computeGradient(s)
    _residual_initial!(s.status, s.x, s.y, s.g)
    s.status.i = 0

    if s.status.rₐ ≥ s.params.atol² || n > 0 || s.params.nmin > 0
        for s.status.i = 1:nmax
            # apply line search
            _linesearch!(s.x̄, s.F, s.x, - s.Q * s.g)
            # solve!(s.x̄, s.y, s.g, s.x, s.ls)

            # compute Gradient at new solution
            s.ȳ  = s.F(s.x̄)
            computeGradient(s.x̄, s.ḡ, s.∇params)

            # compute residual
            _residual!(s.status, s.x̄, s.ȳ, s.ḡ)

            # println(s.status.i, ", ", s.x̄, ", ", s.ȳ, ", ", s.status.rₐ,", ",  s.status.rᵣ,", ",  s.status.rₛ)
            if check_solver_converged(s.status, s.params) && s.status.i ≥ s.params.nmin && !(n > 0)
                if s.params.nwarn > 0 && s.status.i ≥ s.params.nwarn
                    println("WARNING: BFGS Optimizer took ", s.status.i, " iterations.")
                end
                break
            end

            # δ = x̄ - x
            s.δ .= s.x̄ .- s.x

            # γ = ḡ - g
            s.γ .= s.ḡ .- s.g

            # δγ = δᵀγ
            δγ = s.δ ⋅ s.γ

            # DFP
            # Q = Q - ... + ...
            # s.Q .-= s.Q * s.γ * s.γ' * s.Q / (s.γ' * s.Q * s.γ) .+
            #         s.δ * s.δ' ./ δγ

            # BFGS
            # Q = Q - ... + ...
            s.Q .-= (s.δ * s.γ' * s.Q .+ s.Q * s.γ * s.δ') ./ δγ .-
                    (1 + (s.γ' * s.Q * s.γ) ./ δγ) .* (s.δ * s.δ') ./ δγ

            # update status
            s.status.xₚ .= s.x̄
            s.status.yₚ .= s.ȳ

            # update temporaries
            s.x .= s.x̄
            s.y  = s.ȳ
            s.g .= s.ḡ
        end
    end
end

function solve!(x, s::BFGSOptimizer; kwargs...)
    setInitialConditions!(s, x)
    solve!(s; kwargs...)
    x .= s.x̄
end
