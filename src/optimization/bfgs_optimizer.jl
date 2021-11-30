mutable struct BFGSOptimizer{T,FT,TJ,TS<:LineSearch} <: Optimizer{T}
    x::Vector{T}
    x̄::Vector{T}
    δ::Vector{T}

    g::Vector{T}
    ḡ::Vector{T}
    γ::Vector{T}

    Q::Matrix{T}
    y::T
    ȳ::T

    F::FT
    Jparams::TJ

    ls::TS

    params::NonlinearSolverParameters{T}
    status::NonlinearSolverStatus{T}

    function BFGSOptimizer{T,FT,TJ,TS}(x, y, F, Jparams, line_search) where {T,FT,TJ,TS}
        Q = zeros(T, length(x), length(x))
    
        x = zero(x)
        x̄ = zero(x)
        δ = zero(x)
    
        g = zero(x)
        ḡ = zero(x)
        γ = zero(x)
    
        nls_params = NonlinearSolverParameters(T)
        nls_status = NonlinearSolverStatus{T}(length(x))
    
        new(x, x̄, δ, g, ḡ, γ, Q, zero(y), zero(y), F, Jparams, line_search, nls_params, nls_status)
    end
end


function BFGSOptimizer(x::AbstractVector{T}, y::T, F::Function; J!::Union{Function,Nothing} = nothing, linesearch = Bisection(F)) where {T}
    Jparams = getJacobianParameters(J!, (y,x) -> [F(x)], T, length(x), 1)
    BFGSOptimizer{T,typeof(F),typeof(Jparams),typeof(linesearch)}(x, y, F, Jparams, linesearch)
end


status(solver::BFGSOptimizer) = solver.status
params(solver::BFGSOptimizer) = solver.params

function computeJacobian(s::BFGSOptimizer)
    ∇F = x -> ForwardDiff.gradient(s.F, x)
    s.g .= ∇F(s.x)
end

check_jacobian(s::BFGSOptimizer) = check_jacobian(s.g)
print_jacobian(s::BFGSOptimizer) = print_jacobian(s.g)


function setInitialConditions!(s::BFGSOptimizer{T}, x₀::Vector{T}) where {T}
    s.x .= x₀
    s.y = s.F(s.x)
    computeJacobian(s)
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

    computeJacobian(s)
    _residual_initial!(s.status, s.x, s.y, s.g)
    s.status.i = 0

    ∇F = x -> ForwardDiff.gradient(s.F, x)

    if s.status.rₐ ≥ s.params.atol² || n > 0 || s.params.nmin > 0
        for s.status.i = 1:nmax
            # apply line search
            _linesearch!(s.x̄, s.F, s.x, - s.Q * s.g)
            # solve!(s.x̄, s.y, s.g, s.x, s.ls)

            # compute Jacobian at new solution
            s.ȳ  = s.F(s.x̄)
            s.ḡ .= ∇F(s.x̄)

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
