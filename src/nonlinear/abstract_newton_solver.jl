

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
function linesearch_objective(objective!, jacobian!, cache::NewtonSolverCache)
    function f(α)
        cache.x₁ .= cache.x₀ .+ α .* cache.δx
        objective!(cache.y, cache.x₁)
        l2norm(cache.y)
    end

    function d(α)
        cache.x₁ .= cache.x₀ .+ α .* cache.δx
        objective!(cache.y, cache.x₁)
        jacobian!(cache.J, cache.x₁, objective!)
        -2*dot(cache.y, cache.J, cache.δx)
    end

    TemporaryUnivariateObjective(f, d)
end


abstract type AbstractNewtonSolver{T,AT} <: NonlinearSolver end


@define newton_solver_variables begin
    jacobian::TJ

    linear::TL
    linesearch::TLS

    cache::NewtonSolverCache{T,AT,JT}
    config::Options{T}
    status::TST
end

cache(solver::AbstractNewtonSolver) = solver.cache
config(solver::AbstractNewtonSolver) = solver.config
status(solver::AbstractNewtonSolver) = solver.status
jacobian(solver::AbstractNewtonSolver) = solver.jacobian

compute_jacobian!(s::AbstractNewtonSolver, x, f::Callable) = compute_jacobian!(s, x, f, missing)
compute_jacobian!(s::AbstractNewtonSolver, x, f::Callable, ::Missing) = s.jacobian(s.cache.J, x, f)
compute_jacobian!(s::AbstractNewtonSolver, x, f::Callable, j::Callable) = s.jacobian(s.cache.J, x, j)

check_jacobian(s::AbstractNewtonSolver) = check_jacobian(s.jacobian)
print_jacobian(s::AbstractNewtonSolver) = print_jacobian(s.jacobian)

initialize!(s::AbstractNewtonSolver, x₀::AbstractArray, f) = initialize!(status(s), x₀, f)
update!(s::AbstractNewtonSolver, x₀::AbstractArray) = update!(cache(s), x₀)
