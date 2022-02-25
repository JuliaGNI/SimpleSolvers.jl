

struct NewtonSolverCache{T, AT <: AbstractArray{T}}
    x₀::Vector{T}
    x₁::Vector{T}
    δx::Vector{T}
    
    y₀::Vector{T}
    y₁::Vector{T}
    δy::Vector{T}

    function NewtonSolverCache(x::AT, y::AT) where {T, AT <: AbstractArray{T}}
        new{T,AT}(zero(x), zero(x), zero(x),
                  zero(y), zero(y), zero(y))
    end
end

function update!(cache::NewtonSolverCache, x::AbstractVector)
    cache.x₀ .= x
    cache.x₁ .= x
    cache.δx .= 0
    return cache
end

function initialize!(cache::NewtonSolverCache, x::AbstractVector)
    cache.x₀ .= eltype(x)(NaN)
    cache.x₁ .= x
    cache.δx .= eltype(x)(NaN)
    cache.y₀ .= eltype(x)(NaN)
    cache.y₁ .= eltype(x)(NaN)
    cache.δy .= eltype(x)(NaN)
    return cache
end

"create univariate objective for linesearch algorithm"
function linesearch_objective(objective!, cache::NewtonSolverCache)
    function ls_f(α)
        cache.x₁ .= cache.x₀ .+ α .* cache.δx
        objective!(cache.y₁, cache.x₁)
        l2norm(cache.y₁)
    end

    UnivariateObjective(ls_f, 1.)
end


abstract type AbstractNewtonSolver{T,AT} <: NonlinearSolver end


@define newton_solver_variables begin
    x::AT
    y::AT
    J::Matrix{T}

    F!::FT
    Jparams::TJ

    linear::TL
    linesearch::TS

    cache::NewtonSolverCache{T,AT}
    config::Options{T}
    status::NonlinearSolverStatus{T}
end

config(solver::AbstractNewtonSolver) = solver.config
status(solver::AbstractNewtonSolver) = solver.status

compute_jacobian!(s::AbstractNewtonSolver) = compute_jacobian!(s.J, s.x, s.Jparams)
check_jacobian(s::AbstractNewtonSolver) = check_jacobian(s.J)
print_jacobian(s::AbstractNewtonSolver) = print_jacobian(s.J)

function initialize!(s::AbstractNewtonSolver{T}, x₀::Vector{T}) where {T}
    copyto!(s.x, x₀)
    s.F!(s.y, s.x)
    initialize!(status(s), s.x, s.y)
end
