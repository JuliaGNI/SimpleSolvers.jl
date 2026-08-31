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

`params` may carry an optional `φ₀` field holding the merit at the ``\\alpha = 0``
anchor. Every line search evaluates that anchor first, and it is exactly the residual
the solver has *already* computed at the current iterate, so [`solver_step!`](@ref)
passes it along and saves one `F` evaluation per solver step — the most expensive
single operation for a large residual. A caller who drives `solver_step!` by hand from
a state whose `value` is stale must not supply it.

It may carry an optional `αmax` field too, the caller's ceiling on the step length; see
[`linesearch_αmax`](@ref), which documents why that one has to be per call. That one is read by
the *method* rather than by these closures, but through the same `hasproperty` guard, resolved
from the parameter type at compile time, so supplying neither costs nothing. `params` therefore
has to answer *property* access throughout — a `NamedTuple` or any struct, which is what the
required `x` and `parameters` fields already demanded.
"""
function linesearch_problem(nlp::NonlinearProblem, jacobian::Jacobian{T},
        cache::Union{NonlinearSolverCache{T}, DogLegCache{T}}) where {T}
    # private scratch buffers for the line search (see the docstring for why)
    xₜ = zero(solution(cache))
    yₜ = zero(value(cache))
    # `zero_like`, not `zero`: for a sparse Jacobian the latter drops the pattern, and this
    # buffer has to be interchangeable with the solver's own for the caller's `DF!`.
    jₜ = zero_like(jacobianmatrix(cache))

    function f(α::Number, params)
        # `hasproperty` is resolved from the parameter type at compile time, so the guard costs
        # nothing and the closure stays usable with the bare `(x, parameters)` form. It is the
        # same guard `linesearch_αmax` reads `params.αmax` through, and it has to be: this
        # closure reaches its other two fields by *property* access (`params.x`,
        # `params.parameters`), so a `params` that only answers `haskey` — a `Dict` — could not
        # work here anyway, while `haskey` on one that answers only `hasproperty` — any struct, and
        # `NullParameters` among them — raises a `MethodError` from inside the merit.
        (iszero(α) && hasproperty(params, :φ₀)) && return convert(T, params.φ₀)
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
    We apply `L2norm` to the output of `value!` (the evaluation of the nonlinear problem). This is because the solver operates on a function with array-valued outputs from which we have to find roots (in contrast to an optimizer which operates on a function with a scalar output of which we should find a minimum).

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
