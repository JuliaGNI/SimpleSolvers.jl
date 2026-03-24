@doc raw"""
    linesearch_problem(problem, cache)

Create [`LinesearchProblem`](@ref) for linesearch algorithm.

The variable on which this problem depends is ``\alpha``.

# Example

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: NewtonOptimizerCache, linesearch_problem, update!, compute_direction)
julia> x = [1, 0., 0.]
3-element Vector{Float64}:
 1.0
 0.0
 0.0

julia> f = x -> sum(x .^ 3 / 6 + x .^ 2 / 2);

julia> obj = OptimizerProblem(f, x);

julia> grad = GradientAutodiff{Float64}(obj.F, length(x));

julia> hess = HessianAutodiff{Float64}(obj.F, length(x));

julia> cache = NewtonOptimizerCache(x);

julia> state = NewtonOptimizerState(x); update!(state, grad, x);

julia> params = (x = state.x,);

julia> update!(cache, state, grad, hess, x);

julia> ls_obj = linesearch_problem(obj, grad, cache);

julia> ls_obj.F(0., params)
0.6666666666666666

julia> ls_obj.D(0., params)
-1.125

```

!!! info
    Note that in the example above calling [`update!`](@ref) on the [`NewtonOptimizerCache`](@ref) requires a [`Hessian`](@ref).
"""
function linesearch_problem(problem::OptimizerProblem{T}, gradient_instance::Gradient, cache::OptimizerCache{T}) where {T}
    function f(α, params)
        compute_new_iterate!(solution(cache), params.x, α, direction(cache))
        value(problem, solution(cache))
    end

    function d(α, params)
        compute_new_iterate!(solution(cache), params.x, α, direction(cache))
        gradient_instance(gradient(cache), solution(cache))
        dot(gradient(cache), direction(cache))
    end

    LinesearchProblem{T}(f, d)
end
