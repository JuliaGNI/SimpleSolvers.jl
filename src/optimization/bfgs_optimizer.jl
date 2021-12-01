mutable struct BFGSOptimizer{XT, YT, FT <: Callable, T∇ <: GradientParameters, TS <: LineSearch, VT <: AbstractVector{XT}} <: Optimizer{XT}
    x̃::VT
    ỹ::YT
    g̃::VT

    Q::Matrix{XT}
    Q1::Matrix{XT}
    Q2::Matrix{XT}
    Q3::Matrix{XT}
    δγ::Matrix{XT}
    δδ::Matrix{XT}
    
    F::FT
    ∇params::T∇

    ls::TS

    params::NonlinearSolverParameters{XT}
    status::OptimizerStatus{XT,YT,VT}

    function BFGSOptimizer{XT,YT,FT,T∇,TS,VT}(x, F, ∇params, line_search) where {XT,YT,FT,T∇,TS,VT}
        Q = zeros(XT, length(x), length(x))

        Q1 = zero(Q)
        Q2 = zero(Q)
        Q3 = zero(Q)
        δγ = zero(Q)
        δδ = zero(Q)
        
        x̃ = zero(x)
        g̃ = zero(x)

        params = NonlinearSolverParameters(XT)
        status = OptimizerStatus{XT,YT,VT}(length(x))
    
        new(x̃, zero(YT), g̃, Q, Q1, Q2, Q3, δγ, δδ, F, ∇params, line_search, params, status)
    end
end

function BFGSOptimizer(x::VT, F::Function; ∇F!::Union{Callable,Nothing} = nothing, linesearch = Bisection(F)) where {XT, VT <: AbstractVector{XT}}
    ∇params = getGradientParameters(∇F!, F, x)
    YT = typeof(F(x))
    BFGSOptimizer{XT,YT,typeof(F),typeof(∇params),typeof(linesearch),VT}(x, F, ∇params, linesearch)
end


status(s::BFGSOptimizer) = s.status
params(s::BFGSOptimizer) = s.params

check_gradient(s::BFGSOptimizer) = check_gradient(s.g)
print_gradient(s::BFGSOptimizer) = print_gradient(s.g)

print_solver_status(s::BFGSOptimizer) = print_solver_status(status(s), params(s))
check_solver_converged(s::BFGSOptimizer) = check_solver_converged(status(s), params(s))


function setInitialConditions!(s::BFGSOptimizer{T}, x₀::Vector{T}) where {T}
    s.Q .= Matrix(1.0I, size(s.Q)...)
    s.x̃ .= x₀
    s.ỹ  = s.F(s.x̃)
    computeGradient(s.x̃, s.g̃, s.∇params)
    initialize!(status(s), s.x̃, s.ỹ, s.g̃)
end


function _f(s, α)
    s.x̃ .= s.status.x .+ α .* s.status.δ
    s.F(s.x̃)
end

function _linesearch!(s::BFGSOptimizer)
    mul!(s.status.δ, s.Q, s.status.g)
    s.status.δ .*= -1
    objective = α -> _f(s, α)
    a, b = bracket_minimum(objective, 1.0)
    α, s.status.y = s.ls(objective, a, b)
    s.status.x .= s.status.x̄ .+ α .* s.status.δ
end


function solve!(s::BFGSOptimizer{T}; n::Int = 0) where {T}
    local nmax::Int = n > 0 ? nmax = n : s.params.nmax

    s.status.i = 0
    if s.status.rgₐ ≥ s.params.atol² || n > 0 || s.params.nmin > 0
        for s.status.i = 1:nmax
            # apply line search
            _linesearch!(s)

            # compute Gradient at new solution
            computeGradient(s.status.x, s.status.g, s.∇params)

            # compute residual
            residual!(s.status)

            # println(s.status.i, ", ", s.x̄, ", ", s.ȳ, ", ", s.status.rₐ,", ",  s.status.rᵣ,", ",  s.status.rₛ)
            if check_solver_converged(s.status, s.params) && s.status.i ≥ s.params.nmin && !(n > 0)
                if s.params.nwarn > 0 && s.status.i ≥ s.params.nwarn
                    println("WARNING: BFGS Optimizer took ", s.status.i, " iterations.")
                end
                break
            end

            # δ = x - x̄  (already computed in residual)
            # s.status.δ .= s.status.x .- s.status.x̄

            # γ = g - ḡ  (already computed in residual)
            # s.status.γ .= s.status.g .- s.status.ḡ

            # δγ = δᵀγ
            δγ = s.status.δ ⋅ s.status.γ

            # DFP
            # Q = Q - ... + ...
            # s.Q .-= s.Q * s.γ * s.γ' * s.Q / (s.γ' * s.Q * s.γ) .+
            #         s.δ * s.δ' ./ δγ

            # BFGS
            # Q = Q - ... + ...
            # s.Q .-= (s.δ * s.γ' * s.Q .+ s.Q * s.γ * s.δ') ./ δγ .-
            #         (1 + dot(s.γ, s.Q, s.γ) ./ δγ) .* (s.δ * s.δ') ./ δγ

            outer!(s.δγ, s.status.δ, s.status.γ)
            outer!(s.δδ, s.status.δ, s.status.δ)
            mul!(s.Q1, s.δγ, s.Q)
            mul!(s.Q2, s.Q, s.δγ')
            s.Q3 .= (1 + dot(s.status.γ, s.Q, s.status.γ) ./ δγ) .* s.δδ
            s.Q .-= (s.Q1 .+ s.Q2 .- s.Q3) ./ δγ

            # update status and temporaries
            update!(status(s))
        end
    end
end

function solve!(x, s::BFGSOptimizer; kwargs...)
    setInitialConditions!(s, x)
    solve!(s; kwargs...)
    x .= s.params.x
end
