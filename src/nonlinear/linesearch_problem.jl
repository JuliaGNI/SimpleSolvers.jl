"""
    linesearch_problem(nlp, cache, params)

Make a line search problem for a *Newton solver* (the `cache` here is an instance of [`NonlinearSolverCache`](@ref)).

# Implementation

!!! info "Producing a single-valued output"
    Different from the `linesearch_problem` for `NewtonOptimizerCache`s, we apply `L2norm` to the output of `problem!`. This is because the solver operates on an optimizer problem with multiple outputs from which we have to find roots, whereas an optimizer operates on an optimizer problem with a single output of which we should find a minimum.

Also see [`linesearch_problem(::OptimizerProblem{T}, ::Gradient, ::OptimizerCache{T}, ::OptimizerState) where {T}`](@ref).
"""
function linesearch_problem(nlp::NonlinearProblem{T}, jacobian_instance::Jacobian{T}, cache::NonlinearSolverCache{T}, x::AbstractVector{T}, params) where {T}
    function f(α)
        compute_new_iterate!(solution(cache), x, α, direction(cache))
        value!(value(cache), nlp, solution(cache), params)
        L2norm(value(cache))
    end

    function d(α)
        compute_new_iterate!(solution(cache), x, α, direction(cache))
        value!(value(cache), nlp, solution(cache), params)
        jacobian_instance(jacobian(cache), solution(cache), params)
        2dot(value(cache), jacobian(cache), direction(cache))
    end

    # the last argument is to specify the "type" in the problem
    LinesearchProblem(f, d, zero(T))
end

"""
    linesearch_problem(nl::NonlinearSolver, state, params)

Build a line search problem based on a [`NonlinearSolver`](@ref) (almost always a [`NewtonSolver`](@ref) in practice).
"""
linesearch_problem(nl::NonlinearSolver, x::AbstractVector, params) = linesearch_problem(nonlinearproblem(nl), Jacobian(nl), cache(nl), x, params)
