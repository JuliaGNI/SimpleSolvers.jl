"""
    linesearch_problem(nlp, cache, params)

Make a line search problem for a *Newton solver* (the `cache` here is an instance of [`NewtonSolverCache`](@ref)).

# Implementation

!!! info "Producing a single-valued output"
    Different from the `linesearch_problem` for `NewtonOptimizerCache`s, we apply `l2norm` to the output of `problem!`. This is because the solver operates on an optimizer problem with multiple outputs from which we have to find roots, whereas an optimizer operates on an optimizer problem with a single output of which we should find a minimum.

Also see [`linesearch_problem(::OptimizerProblem{T}, ::Gradient, ::OptimizerCache{T}, ::OptimizerState) where {T}`](@ref).
"""
function linesearch_problem(nlp::NonlinearProblem{T}, jacobian_instance::Jacobian{T}, cache::NewtonSolverCache{T}, params) where {T}
    function f(α)
        cache.x .= compute_new_iterate(cache.x̄, α, cache.δx)
        value!(nlp, cache.x, params)
        cache.y .= value(nlp)
        L2norm(cache.y)
    end

    function d(α)
        cache.x .= compute_new_iterate(cache.x̄, α, cache.δx)
        value!(nlp, cache.x, params)
        cache.y .= value(nlp)
        jacobian_instance(nlp, cache.x, params)
        2 * dot(cache.y, jacobian(nlp), direction(cache))
    end

    # the last argument is to specify the "type" in the problem
    LinesearchProblem(f, d, zero(T))
end

"""
    linesearch_problem(nl::NonlinearSolver, params)

Build a line search problem based on a [`NonlinearSolver`](@ref) (almost always a [`NewtonSolver`](@ref) in practice).
"""
linesearch_problem(nl::NonlinearSolver, params) = linesearch_problem(nonlinearproblem(nl), Jacobian(nl), cache(nl), params)
