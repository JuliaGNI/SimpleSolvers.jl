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
        cache.x .= compute_new_iterate(cache.x̄, α, cache.δx)
        value!(objective, cache.x)
        cache.y .= value(objective)
        L2norm(cache.y)
    end

    function d(α)
        cache.x .= compute_new_iterate(cache.x̄, α, cache.δx)
        value!(objective, cache.x)
        cache.y .= value(objective)
        compute_jacobian!(cache.J, cache.x, jacobian!)
        2 * dot(cache.y, cache.J, cache.δx)
    end

    # the last argument is to specify the "type" in the objective
    TemporaryUnivariateObjective(f, d, zero(T))
end