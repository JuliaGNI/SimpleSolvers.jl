

struct NewtonSolverCache{T, AT <: AbstractVector{T}, JT <: AbstractMatrix{T}}
    x₀::AT
    x₁::AT
    δx::AT

    rhs::AT
    y::AT
    J::JT

    function NewtonSolverCache(x::AT, y::AT) where {T, AT <: AbstractArray{T}}
        J = alloc_j(x,y)
        new{T, AT, typeof(J)}(zero(x), zero(x), zero(x), zero(y), zero(y), J)
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

    cache.rhs .= eltype(x)(NaN)
    cache.y .= eltype(x)(NaN)
    cache.J .= eltype(x)(NaN)

    return cache
end

"create univariate objective for linesearch algorithm"
function linesearch_objective(objective!, jacobian!, cache::NewtonSolverCache{T}) where {T}
    function f(α)
        cache.x₁ .= cache.x₀ .+ α .* cache.δx
        objective!(cache.y, cache.x₁)
        l2norm(cache.y)
    end

    function d(α)
        cache.x₁ .= cache.x₀ .+ α .* cache.δx
        objective!(cache.y, cache.x₁)
        jacobian!(cache.J, cache.x₁)
        -2*dot(cache.y, cache.J, cache.δx)
    end

    UnivariateObjective(f, d, one(T))
end


abstract type AbstractNewtonSolver{T,AT} <: NonlinearSolver end


@define newton_solver_variables begin
    F!::FT
    J!::TJ

    linear::TL
    linesearch::TS

    cache::NewtonSolverCache{T,AT}
    config::Options{T}
    status::NonlinearSolverStatus{T}
end

cache(solver::AbstractNewtonSolver) = solver.cache
config(solver::AbstractNewtonSolver) = solver.config
status(solver::AbstractNewtonSolver) = solver.status

compute_jacobian!(s::AbstractNewtonSolver, x) = compute_jacobian!(s.cache.J, x, s.J!)
check_jacobian(s::AbstractNewtonSolver) = check_jacobian(s.J)
print_jacobian(s::AbstractNewtonSolver) = print_jacobian(s.J)

initialize!(s::AbstractNewtonSolver, x₀::AbstractArray) = initialize!(status(s), x₀)
update!(s::AbstractNewtonSolver, x₀::AbstractArray) = update!(cache(s), x₀)
