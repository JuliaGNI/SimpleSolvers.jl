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

jacobian(cache::NewtonSolverCache) = cache.J
direction(cache::NewtonSolverCache) = cache.δx

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

"""
    linesearch_objective(objective!, jacobian!, cache)

Make a line search objective for a *Newton solver* (the `cache` here is an instance of [`NewtonSolverCache`](@ref)).

# Implementation

!!! info "Producing a single-valued output"
    Different from the `linesearch_objective` for `NewtonOptimizerCache`s, we apply `l2norm` to the output of `objective!`. This is because the solver operates on an objective with multiple outputs from which we have to find roots, whereas an optimizer operates on an objective with a single output of which we should find a minimum.

Also see [`linesearch_objective(::MultivariateObjective{T}, ::NewtonOptimizerCache{T}) where {T}`](@ref).
"""
function linesearch_objective(objective::AbstractObjective, jacobian!::Jacobian, cache::NewtonSolverCache{T}) where T
    function f(α)
        cache.x₁ .= compute_new_iterate(cache.x₀, α, cache.δx)
        value!(objective, cache.x₁)
        cache.y .= value(objective)
        L2norm(cache.y)
    end

    function d(α)
        cache.x₁ .= compute_new_iterate(cache.x₀, α, cache.δx)
        value!(objective, cache.x₁)
        cache.y .= value(objective)
        compute_jacobian!(cache.J, cache.x₁, jacobian!)
        2 * dot(cache.y, cache.J, cache.δx)
    end

    # the last argument is to specify the "type" in the objective
    TemporaryUnivariateObjective(f, d, zero(T))
end

"""
    AbstractNewtonSolver <: NonlinearSolver

A supertype that comprises e.g. [`NewtonSolver`](@ref).
"""
abstract type AbstractNewtonSolver{T,AT} <: NonlinearSolver end

cache(solver::AbstractNewtonSolver) = solver.cache
config(solver::AbstractNewtonSolver) = solver.config
status(solver::AbstractNewtonSolver) = solver.status

"""
    jacobian(solver::AbstractNewtonSolver)

Calling `jacobian` on an instance of [`AbstractNewtonSolver`](@ref) produces a slight ambiguity since the `cache` (of type [`NewtonSolverCache`](@ref)) also stores a Jacobian, but in the latter case it is a matrix not an instance of type [`Jacobian`](@ref).
Hence we return the object of type [`Jacobian`](@ref) when calling `jacobian`. This is also used in [`solver_step!`](@ref).
"""
jacobian(solver::AbstractNewtonSolver)::Jacobian = solver.jacobian

"""
    linearsolver(solver)

Return the linear part (i.e. a [`LinearSolver`](@ref)) of an [`AbstractNewtonSolver`](@ref).

# Examples

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: linearsolver)
x = rand(3)
y = rand(3)
F(y, x) = y .= tanh.(x)
s = NewtonSolver(x, y; F = F)
linearsolver(s)

# output

LUSolver{Float64}(3, [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0], [1, 2, 3], [1, 2, 3], 1)
```
"""
linearsolver(solver::AbstractNewtonSolver) = solver.linear
linesearch(solver::AbstractNewtonSolver) = solver.linesearch

# compute_jacobian!(s::AbstractNewtonSolver, x) = compute_jacobian!(jacobian(s), x, f)
compute_jacobian!(s::AbstractNewtonSolver, x, jacobian!::Union{Jacobian, Callable}; kwargs...) = compute_jacobian!(jacobian(cache(s)), x, jacobian!; kwargs...)
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