"""
    linesearch_problem(nls, cache, params)

Make a line search problem for a *Newton solver* (the `cache` here is an instance of [`NewtonSolverCache`](@ref)).

# Implementation

!!! info "Producing a single-valued output"
    Different from the `linesearch_problem` for `NewtonOptimizerCache`s, we apply `l2norm` to the output of `objective!`. This is because the solver operates on an objective with multiple outputs from which we have to find roots, whereas an optimizer operates on an objective with a single output of which we should find a minimum.

Also see [`linesearch_problem(::OptimizerProblem{T}, ::NewtonOptimizerCache{T}) where {T}`](@ref).
"""
function linesearch_problem(nls::NonlinearProblem{T}, cache::NewtonSolverCache{T}, params) where {T}
    function f(α)
        cache.x .= compute_new_iterate(cache.x̄, α, cache.δx)
        value!(nls, cache.x, params)
        cache.y .= value(nls)
        L2norm(cache.y)
    end

    function d(α)
        cache.x .= compute_new_iterate(cache.x̄, α, cache.δx)
        value!(nls, cache.x, params)
        cache.y .= value(nls)
        jacobian!(nls, cache.x, params)
        2 * dot(cache.y, jacobian(nls), direction(cache))
    end

    # the last argument is to specify the "type" in the objective
    LinesearchProblem(f, d, zero(T))
end

"""
    linesearch_problem(nl::NonlinearSolver, params)

Build a line search problem based on a [`NonlinearSolver`](@ref) (almost always a [`NewtonSolver`](@ref) in practice).
"""
linesearch_problem(nl::NonlinearSolver, params) = linesearch_problem(nonlinearproblem(nl), cache(nl), params)
