@doc raw"""
    linesearch_problem(problem, cache)

Create [`LinesearchProblem`](@ref) for linesearch algorithm. The variable on which this problem depends is ``\alpha``.

# Example

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: NewtonOptimizerCache, linesearch_problem, update!)
x = [1, 0., 0.]
f = x -> sum(x .^ 3 / 6 + x .^ 2 / 2)
obj = OptimizerProblem(f, x)
grad = GradientAutodiff{Float64}(obj.F, length(x))
value!(obj, x)
cache = NewtonOptimizerCache(x)
hess = HessianAutodiff(obj, x)
update!(hess, x)
update!(cache, obj, grad, hess, x)
x₂ = [.9, 0., 0.]
value!(obj, x₂)
update!(hess, x₂)
update!(cache, obj, grad, hess, x₂)
ls_obj = linesearch_problem(obj, grad, cache)
α = .1
(ls_obj.F(α), ls_obj.D(α))

# output

(0.4412947468016475, -0.8083161485821551)
```

In the example above we have to apply [`update!`](@ref) twice on the instance of [`NewtonOptimizerCache`](@ref) because it needs to store the current *and* the previous iterate.

# Implementation

Calling the function and derivative stored in the [`LinesearchProblem`](@ref) created with `linesearch_problem` does not allocate a new array, but uses the one stored in the instance of [`NewtonOptimizerCache`](@ref).
"""
function linesearch_problem(problem::OptimizerProblem{T}, gradient_instance::Gradient, cache::NewtonOptimizerCache{T}) where {T}
    function f(α)
        cache.x .= compute_new_iterate(cache.x̄, α, direction(cache))
        value!(problem, cache.x)
    end

    function d(α)
        cache.x .= compute_new_iterate(cache.x̄, α, direction(cache))
        gradient!(problem, gradient_instance, cache.x)
        cache.g .= problem.g
        dot(cache.g, direction(cache))
    end

    LinesearchProblem{T}(f, d)
end

