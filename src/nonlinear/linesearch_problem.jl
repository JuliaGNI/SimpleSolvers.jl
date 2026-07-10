"""
    linesearch_problem(nlp, jacobian, cache)

Make a line search problem for a *Newton solver* (the `cache` here is an instance of [`NonlinearSolverCache`](@ref)).

# Extended help

The line search closures evaluate the merit at trial steps `α` using *private*
scratch buffers rather than the solver's shared `cache`. The shared buffers
(`solution`/`value`/`jacobianmatrix`) are read by the solver *after* the line
search returns (e.g. the next `direction!` step and the convergence check), so
writing trial iterates into them would be an aliasing hazard. The line search
therefore only *reads* the current direction from the shared cache and the
current iterate from `params.x`; every write goes to a closure-owned buffer.
"""
function linesearch_problem(nlp::NonlinearProblem, jacobian::Jacobian{T}, cache::Union{NonlinearSolverCache{T},DogLegCache{T}}) where {T}
    # private scratch buffers for the line search (see the docstring for why)
    xₜ = zero(solution(cache))
    yₜ = zero(value(cache))
    jₜ = zero(jacobianmatrix(cache))

    function f(α::Number, params)
        compute_new_iterate!(xₜ, params.x, α, direction(cache))
        value!(yₜ, nlp, xₜ, params.parameters)
        L2norm(yₜ)
    end

    function d(α::Number, params)
        compute_new_iterate!(xₜ, params.x, α, direction(cache))
        value!(yₜ, nlp, xₜ, params.parameters)
        jacobian(jₜ, xₜ, params.parameters)
        2dot(yₜ, jₜ, direction(cache))
    end

    LinesearchProblem{T}(f, d)
end

@doc raw"""
    linesearch_problem(nl::NonlinearSolver)

Build a line search problem based on a [`NonlinearSolver`](@ref).

!!! info "Producing a single-valued output"
    Different from the `linesearch_problem` for `NewtonOptimizerCache`s, we apply `L2norm` to the output of `problem!`. This is because the solver operates on an optimizer problem with multiple outputs from which we have to find roots, whereas an optimizer operates on an optimizer problem with a single output of which we should find a minimum.

# Examples

We show how to set up the [`LinesearchProblem`](@ref) for a simple example and compute ``f^\mathrm{ls}(\alpha_0)`` and ``\partial{}f^\mathrm{ls}/\partial\alpha(\alpha_0)``.

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: linesearch_problem, NullParameters, direction!)
julia> F(y, x, params) = y .= (x .- 1.).^2;

julia> x = ones(3)/2; y = similar(x); nl = NewtonSolver(x, y; F = F);

julia> _params = NullParameters();

julia> direction!(nl, x, _params, 1)
3-element Vector{Float64}:
 0.25
 0.25
 0.25

julia> ls_prob = linesearch_problem(nl);

julia> state = NonlinearSolverState(x); update!(state, x, F(y, x, _params));

julia> params = (parameters = _params, x = state.x)
(parameters = NullParameters(), x = [0.5, 0.5, 0.5])

julia> ls_prob.F(0., params)
0.1875

julia> ls_prob.D(0., params)
-0.375
```
"""
linesearch_problem(nl::NonlinearSolver) = linesearch_problem(nonlinearproblem(nl), jacobian(nl), cache(nl))
