"""
    NewtonSolverCache

Stores `x₀`, `x₁`, `δx`, `rhs`, `y` and `J`.

# Keys

`δx`: search direction.

# Constructor

```julia
NewtonSolverCache(x, y)
```

`J` is allocated by calling [`alloc_j`](@ref).
"""
struct NewtonSolverCache{T, AT <: AbstractVector{T}, JT <: AbstractMatrix{T}}
    x₀::AT
    x₁::AT
    δx::AT

    rhs::AT
    y::AT
    J::JT

    function NewtonSolverCache(x::AT, y::AT) where {T,AT<:AbstractArray{T}}
        J = alloc_j(x, y)
        c = new{T,AT,typeof(J)}(zero(x), zero(x), zero(x), zero(y), zero(y), J)
        initialize!(c, fill!(similar(x), NaN))
        c
    end
end

@doc raw"""
    update!(cache, x)

Update the [`NewtonSolverCache`](@ref) based on `x`, i.e.:
1. `cache.x₀` ``\gets`` x,
2. `cache.x₁` ``\gets`` x,
3. `cache.δx` ``\gets`` 0.
"""
function update!(cache::NewtonSolverCache, x::AbstractVector)
    cache.x₀ .= x
    cache.x₁ .= x
    cache.δx .= 0
    
    cache
end

"""
    initialize!(cache, x)

Initialize the [`NewtonSolverCache`](@ref) based on `x`.

# Implementation

This calls [`alloc_x`](@ref) to do all the initialization.
"""
function initialize!(cache::NewtonSolverCache, x::AbstractVector)
    cache.x₀ .= alloc_x(x)
    cache.x₁ .= alloc_x(x)
    cache.δx .= alloc_x(x)

    cache.rhs .= alloc_x(x)
    cache.y .= alloc_x(x)
    cache.J .= alloc_x(x)

    cache
end

function linesearch_objective(objective!::Callable, jacobian!::Jacobian, cache::NewtonSolverCache)
    function f(α)
        cache.x₁ .= compute_new_iterate(cache.x₀, α, cache.δx)
        objective!(cache.y, cache.x₁)
        l2norm(cache.y)
    end

    function d(α)
        cache.x₁ .= compute_new_iterate(cache.x₀, α, cache.δx)
        objective!(cache.y, cache.x₁)
        jacobian!(cache.J, cache.x₁, objective!)
        2*dot(cache.y, cache.J, cache.δx)
    end

    TemporaryUnivariateObjective(f, d)
end

linesearch_objective(objective!::AbstractObjective, jacobian!::Jacobian, cache::NewtonSolverCache) = linesearch_objective(objective!.F, jacobian!, cache)

"""
    AbstractNewtonSolver <: NonlinearSolver

A supertype that comprises e.g. [`NewtonSolver`](@ref).
"""
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
linearsolver(solver::AbstractNewtonSolver) = solver.linear
linesearch(solver::AbstractNewtonSolver) = solver.linesearch

compute_jacobian!(s::AbstractNewtonSolver, x) = compute_jacobian!(s.cache.J, x, f)
compute_jacobian!(s::AbstractNewtonSolver, x, jacobian!::Jacobian) = compute_jacobian!(s.cache.J, x, jacobian!)
compute_jacobian!(s::AbstractNewtonSolver, x, jacobian!::Callable) = compute_jacobian!(s.cahce.J, x, jacobian!)
# compute_jacobian!(s::AbstractNewtonSolver, x, f::Callable) = compute_jacobian!(s, x, f, missing)
# compute_jacobian!(s::AbstractNewtonSolver, x, f::Callable, ::Missing) = s.jacobian(s.cache.J, x, f)
# compute_jacobian!(s::AbstractNewtonSolver, x, f::Callable, j::Callable) = s.jacobian(s.cache.J, x, j)

check_jacobian(s::AbstractNewtonSolver) = check_jacobian(jacobian(s))
print_jacobian(s::AbstractNewtonSolver) = print_jacobian(jacobian(s))

initialize!(s::AbstractNewtonSolver, x₀::AbstractArray, f) = initialize!(status(s), x₀, f)
update!(s::AbstractNewtonSolver, x₀::AbstractArray) = update!(cache(s), x₀)

function solve!(x, f::Callable, s::AbstractNewtonSolver)
    solve!(x, f, jacobian(s), s)
end