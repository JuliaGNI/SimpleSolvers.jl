@doc raw"""
    linesearch_problem(problem, cache)

Create [`LinesearchProblem`](@ref) for linesearch algorithm. The variable on which this problem depends is ``\alpha``.

# Example

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: NewtonOptimizerCache, linesearch_problem, update!, compute_direction)
x = [1, 0., 0.]
f = x -> sum(x .^ 3 / 6 + x .^ 2 / 2)
obj = OptimizerProblem(f, x)
grad = GradientAutodiff{Float64}(obj.F, length(x))
cache = NewtonOptimizerCache(x)
state = NewtonOptimizerState(x)
hess = HessianAutodiff(obj, x)
update!(state, grad, x)
update!(cache, state, grad, hess, x)
x₂ = [.9, 0., 0.]
update!(state, grad, x₂)
update!(cache, state, grad, hess, x₂)
compute_direction(cache)
ls_obj = linesearch_problem(obj, grad, cache, state)
α = .1
(ls_obj.F(α), ls_obj.D(α))
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
ls_obj = linesearch_problem(obj, grad, cache, state)
α = .1
(ls_obj.F(α), ls_obj.D(α))

# output

(0.5683038684544637, -0.9375328383328476)
```

In the example above we have to apply [`update!`](@ref) twice on the instance of [`NewtonOptimizerCache`](@ref) because it needs to store the current *and* the previous iterate.

# Implementation

Calling the function and derivative stored in the [`LinesearchProblem`](@ref) created with `linesearch_problem` does not allocate a new array, but uses the one stored in the instance of [`NewtonOptimizerCache`](@ref).
"""
function linesearch_problem(problem::OptimizerProblem{T}, gradient_instance::Gradient, cache::OptimizerCache{T}, state::OptimizerState) where {T}
    function f(α)
        compute_new_iterate!(cache.x, state.x̄, α, direction(cache))
        value(problem, cache.x)
    end

    function d(α)
        compute_new_iterate!(cache.x, state.x̄, α, direction(cache))
        gradient_instance(cache.g, cache.x)
        dot(cache.g, direction(cache))
    end

    LinesearchProblem{T}(f, d)
end

# this is only included now and should be removed later!!!
function linesearch_problem(problem::OptimizerProblem{T}, gradient_instance::Gradient, cache::OptimizerCache{T}, state::NewtonOptimizerState) where {T}
    function f(α)
        compute_new_iterate!(cache.x, state.x, α, direction(cache))
        value(problem, cache.x)
    end

    function d(α)
        compute_new_iterate!(cache.x, state.x, α, direction(cache))
        gradient_instance(cache.g, cache.x)
        dot(cache.g, direction(cache))
    end

    LinesearchProblem{T}(f, d)
end
