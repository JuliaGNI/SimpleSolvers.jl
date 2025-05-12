@doc raw"""
    linesearch_objective(objective, cache)

Create [`TemporaryUnivariateObjective`](@ref) for linesearch algorithm. The variable on which this objective depends is ``\alpha``.

# Example

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: NewtonOptimizerCache, linesearch_objective, update!)
x = [1, 0., 0.]
f = x -> sum(x .^ 3 / 6 + x .^ 2 / 2)
obj = MultivariateObjective(f, x)
gradient!(obj, x)
value!(obj, x)
cache = NewtonOptimizerCache(x)
hess = Hessian(obj, x; mode = :autodiff)
update!(hess, x)
update!(cache, x, obj.g, hess)
x₂ = [.9, 0., 0.]
gradient!(obj, x₂)
value!(obj, x₂)
update!(hess, x₂)
update!(cache, x₂, obj.g, hess)
ls_obj = linesearch_objective(obj, cache)
α = .1
(ls_obj.F(α), ls_obj.D(α))

# output

(0.4412947468016475, -0.8083161485821551)
```

In the example above we have to apply [`update!`](@ref) twice on the instance of [`NewtonOptimizerCache`](@ref) because it needs to store the current *and* the previous iterate.

# Implementation

Calling the function and derivative stored in the [`TemporaryUnivariateObjective`](@ref) created with `linesearch_objective` does not allocate a new array, but uses the one stored in the instance of [`NewtonOptimizerCache`](@ref).
"""
function linesearch_objective(objective::MultivariateObjective{T}, cache::NewtonOptimizerCache{T}) where {T}
    function f(α)
        cache.x .= compute_new_iterate(cache.x̄, α, direction(cache))
        value!(objective, cache.x)
    end

    function d(α)
        cache.x .= compute_new_iterate(cache.x̄, α, direction(cache))
        gradient!(objective, cache.x)
        cache.g .= objective.g
        dot(cache.g, direction(cache))
    end

    TemporaryUnivariateObjective{T}(f, d)
end

