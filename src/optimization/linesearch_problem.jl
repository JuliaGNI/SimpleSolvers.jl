@doc raw"""
    linesearch_problem(problem, cache)

Create [`LinesearchProblem`](@ref) for linesearch algorithm. The variable on which this problem depends is ``\alpha``.

# Example

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: NewtonOptimizerCache, linesearch_problem, update!)
x = [1, 0., 0.]
f = x -> sum(x .^ 3 / 6 + x .^ 2 / 2)
obj = OptimizerProblem(f, x)
grad = GradientAutodiff{Float64}(obj.F, length(x))
cache = NewtonOptimizerCache(x)
state = NewtonOptimizerState(x)
params = (x = state.x̄,)
state.x̄ .= x
hess = HessianAutodiff(obj, x)
update!(cache, state, grad, hess, x)
x₂ = [.9, 0., 0.]
update!(cache, state, grad, hess, x₂)
ls_obj = linesearch_problem(obj, grad, cache)
α = .1
(ls_obj.F(α, params), ls_obj.D(α, params))
x = [1, 0., 0.]
f = x -> sum(x .^ 3 / 6 + x .^ 2 / 2)
obj = OptimizerProblem(f, x)
grad = GradientAutodiff{Float64}(obj.F, length(x))
cache = NewtonOptimizerCache(x)
state = NewtonOptimizerState(x)
state.x̄ .= x
hess = HessianAutodiff(obj, x)
update!(cache, state, grad, hess, x)
x₂ = [.9, 0., 0.]
update!(cache, state, grad, hess, x₂)
ls_obj = linesearch_problem(obj, grad, cache)
α = .1
(ls_obj.F(α, params), ls_obj.D(α, params))

# output

(0.5683038684544637, -0.9375328383328476)
```

In the example above we have to apply [`update!`](@ref) twice on the instance of [`NewtonOptimizerCache`](@ref) because it needs to store the current *and* the previous iterate.

# Implementation

Calling the function and derivative stored in the [`LinesearchProblem`](@ref) created with `linesearch_problem` does not allocate a new array, but uses the one stored in the instance of [`NewtonOptimizerCache`](@ref).
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
