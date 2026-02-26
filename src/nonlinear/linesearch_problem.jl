"""
    linesearch_problem(nlp, cache, params)

Make a line search problem for a *Newton solver* (the `cache` here is an instance of [`NonlinearSolverCache`](@ref)).

# Implementation

!!! info "Producing a single-valued output"
    Different from the `linesearch_problem` for `NewtonOptimizerCache`s, we apply `L2norm` to the output of `problem!`. This is because the solver operates on an optimizer problem with multiple outputs from which we have to find roots, whereas an optimizer operates on an optimizer problem with a single output of which we should find a minimum.

Also see [`linesearch_problem(::OptimizerProblem{T}, ::Gradient, ::OptimizerCache{T}) where {T}`](@ref).
"""
function linesearch_problem(nlp::NonlinearProblem{T}, jacobian::Jacobian{T}, cache::Union{NonlinearSolverCache{T},DogLegCache{T}}) where {T}
    function f(α::Number, params)
        compute_new_iterate!(solution(cache), params.x, α, direction(cache))
        value!(value(cache), nlp, solution(cache), params.parameters)
        L2norm(value(cache))
    end

    function d(α::Number, params)
        compute_new_iterate!(solution(cache), params.x, α, direction(cache))
        value!(value(cache), nlp, solution(cache), params.parameters)
        jacobian(jacobianmatrix(cache), solution(cache), params.parameters)
        2dot(value(cache), jacobianmatrix(cache), direction(cache))
    end

    LinesearchProblem{T}(f, d)
end

"""
    linesearch_problem(nl::NonlinearSolver, state, params)

Build a line search problem based on a [`NonlinearSolver`](@ref) (almost always a [`NewtonSolver`](@ref) in practice).
"""
linesearch_problem(nl::NonlinearSolver) = linesearch_problem(nonlinearproblem(nl), jacobian(nl), cache(nl))
