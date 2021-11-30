mutable struct BFGSOptimizer{XT, YT, FT <: Callable, T∇ <: GradientParameters, TS <: LineSearch, VT <: AbstractVector{XT}} <: Optimizer{XT}
    x::VT
    x̄::VT
    x̃::VT
    δ::VT

    g::VT
    ḡ::VT
    γ::VT

    Q::Matrix{XT}
    Q1::Matrix{XT}
    Q2::Matrix{XT}
    Q3::Matrix{XT}
    δγ::Matrix{XT}
    δδ::Matrix{XT}
    
    y::YT
    ȳ::YT

    F::FT
    ∇params::T∇

    ls::TS

    params::NonlinearSolverParameters{XT}
    status::NonlinearSolverStatus{XT}

    function BFGSOptimizer{XT,YT,FT,T∇,TS,VT}(x, F, ∇params, line_search) where {XT,YT,FT,T∇,TS,VT}
        Q = zeros(XT, length(x), length(x))

        Q1 = zero(Q)
        Q2 = zero(Q)
        Q3 = zero(Q)
        δγ = zero(Q)
        δδ = zero(Q)
        
        x = zero(x)
        x̄ = zero(x)
        x̃ = zero(x)
        δ = zero(x)
    
        g = zero(x)
        ḡ = zero(x)
        γ = zero(x)

        nls_params = NonlinearSolverParameters(XT)
        nls_status = NonlinearSolverStatus{XT}(length(x))
    
        new(x, x̄, x̃, δ, g, ḡ, γ, Q, Q1, Q2, Q3, δγ, δδ, zero(YT), zero(YT), F, ∇params, line_search, nls_params, nls_status)
    end
end


function BFGSOptimizer(x::VT, F::Function; ∇F!::Union{Callable,Nothing} = nothing, linesearch = Bisection(F)) where {XT, VT <: AbstractVector{XT}}
    ∇params = getGradientParameters(∇F!, F, x)
    YT = typeof(F(x))
    BFGSOptimizer{XT,YT,typeof(F),typeof(∇params),typeof(linesearch),VT}(x, F, ∇params, linesearch)
end


status(solver::BFGSOptimizer) = solver.status
params(solver::BFGSOptimizer) = solver.params

check_gradient(s::BFGSOptimizer) = check_gradient(s.g)
print_gradient(s::BFGSOptimizer) = print_gradient(s.g)


function setInitialConditions!(s::BFGSOptimizer{T}, x₀::Vector{T}) where {T}
    s.x .= x₀
    s.y  = s.F(s.x)
    s.Q .= Matrix(1.0I, length(s.x), length(s.x))
    computeGradient(s.x, s.g, s.∇params)
    _residual_initial!(s.status, s.x, s.y, s.g)
end


function update!(s::BFGSOptimizer)
    # update status
    s.status.xₚ .= s.x̄
    s.status.yₚ .= s.ȳ

    # update temporaries
    s.x .= s.x̄
    s.y  = s.ȳ
    s.g .= s.ḡ
end


function _residual_initial!(status, x, y, g)
    status.rₐ = Inf
    status.rᵣ = Inf
    status.rₛ = Inf
    status.r₀ .= 0
    status.x₀ .= x
    status.xₚ .= x
    status.y₀ .= y
    status.yₚ .= y
end


function _residual!(status, x̄, ȳ, ḡ)
    status.rₐ  = norm(status.yₚ[1] - ȳ[1])
    status.rᵣ  = norm(status.yₚ[1] - ȳ[1]) / norm(status.yₚ[1])
    status.r₀ .= status.xₚ .- x̄
    status.rₛ  = norm(status.r₀)
end


function _f(s, α)
    s.x̃ .= s.x .+ α .* s.δ
    s.F(s.x̃)
end

function _linesearch!(s::BFGSOptimizer)
    mul!(s.δ, s.Q, s.g)
    s.δ .*= -1
    objective = α -> _f(s, α)
    a, b = bracket_minimum(objective, 1.0)
    α, y = s.ls(objective, a, b)
    s.x̄ .= s.x .+ α .* s.δ
    s.ȳ  = y
end


function solve!(s::BFGSOptimizer{T}; n::Int = 0) where {T}
    local nmax::Int = n > 0 ? nmax = n : s.params.nmax

    s.status.i = 0
    if s.status.rₐ ≥ s.params.atol² || n > 0 || s.params.nmin > 0
        for s.status.i = 1:nmax
            # apply line search
            _linesearch!(s)

            # compute Gradient at new solution
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
            # s.Q .-= (s.δ * s.γ' * s.Q .+ s.Q * s.γ * s.δ') ./ δγ .-
            #         (1 + dot(s.γ, s.Q, s.γ) ./ δγ) .* (s.δ * s.δ') ./ δγ

            outer!(s.δγ, s.δ, s.γ)
            outer!(s.δδ, s.δ, s.δ)
            mul!(s.Q1, s.δγ, s.Q)
            mul!(s.Q2, s.Q, s.δγ')
            s.Q3 .= (1 + dot(s.γ, s.Q, s.γ) ./ δγ) .* s.δδ
            s.Q .-= (s.Q1 .+ s.Q2 .- s.Q3) ./ δγ

            # update status and temporaries
            update!(s)
        end
    end
end

function solve!(x, s::BFGSOptimizer; kwargs...)
    setInitialConditions!(s, x)
    solve!(s; kwargs...)
    x .= s.x̄
end
