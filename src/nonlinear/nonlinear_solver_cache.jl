"""
    AbstractNonlinearSolverCache

An abstract type that comprises e.g. the [`NonlinearSolverCache`](@ref) and the [`DogLegCache`](@ref).
"""
abstract type AbstractNonlinearSolverCache{T} end

"""
    NonlinearSolverCache <: AbstractNonlinearSolverCache

Derived from [`AbstractNonlinearSolverCache`](@ref). Used in [`NonlinearSolver`](@ref).

# Keys

- `x`: the next iterate (or *guess* thereof),
- `Δx`: search direction. This is updated when calling [`solver_step!`](@ref) via the [`LinearSolver`](@ref) stored in the [`NewtonSolver`](@ref),
- `rhs`: the right-hand-side (this can be accessed by calling [`rhs`](@ref)),
- `y`: the problem evaluated at `x`,
- `j::AbstractMatrix`: the Jacobian evaluated at `x`. Note that this is not of type [`Jacobian`](@ref)!

!!! info
    The line search reads the current search direction `Δx` from this cache but
    writes its trial iterate, residual and Jacobian into its own private buffers
    (see [`linesearch_problem`](@ref)); it does **not** overwrite `x`, `y` or `j`.

"""
struct NonlinearSolverCache{T,AT<:AbstractVector{T},JT<:AbstractMatrix{T}} <: AbstractNonlinearSolverCache{T}
    x::AT
    Δx::AT

    rhs::AT
    y::AT

    j::JT

    # `j` is a *prototype*: its storage and, if sparse, its sparsity pattern are adopted for
    # the whole solve. Pass a `SparseMatrixCSC` to run a sparse Jacobian through the solver;
    # see the `jacobian_prototype` keyword of `NewtonSolver`.
    #
    # A copy, for the same reason `LinearProblem` copies its arguments: `initialize!` below
    # fills the Jacobian with `NaN`s, so keeping the caller's array would destroy the values
    # in the prototype they passed and make two solvers built from one prototype share a
    # Jacobian.
    function NonlinearSolverCache(x::AT, y::AT, jprototype::AbstractMatrix{T}=alloc_j(x, y)) where {T,AT<:AbstractArray{T}}
        j = copy(jprototype)
        c = new{T,AT,typeof(j)}(zero(x), zero(x), zero(y), zero(y), j)
        initialize!(c, fill!(similar(x), NaN))
        c
    end
end

"""
    direction(cache)

Return the direction (i.e. the step vector ``\\Delta{}x``) stored in a solver
cache such as [`NonlinearSolverCache`](@ref) or [`DogLegCache`](@ref).
"""
direction(cache::NonlinearSolverCache) = cache.Δx
jacobianmatrix(cache::NonlinearSolverCache) = cache.j
solution(cache::NonlinearSolverCache) = cache.x
value(cache::NonlinearSolverCache) = cache.y

"""
    rhs(cache)

Return the right-hand side of the equation, stored in `cache::`[`NonlinearSolverCache`](@ref).
"""
rhs(cache::NonlinearSolverCache) = cache.rhs

"""
    initialize!(cache, x)

Initialize the [`NonlinearSolverCache`](@ref) with `NaN`s.
"""
function initialize!(cache::NonlinearSolverCache{T}, ::AbstractVector{T}) where {T}
    solution(cache) .= T(NaN)
    direction(cache) .= T(NaN)

    rhs(cache) .= T(NaN)
    value(cache) .= T(NaN)

    fill_nan!(jacobianmatrix(cache))

    cache
end
