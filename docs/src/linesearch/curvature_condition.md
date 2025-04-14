# The Curvature Condition

The *curvature condition* is used together with the [sufficient decrease condition](@ref "Sufficient Decrease Condition*) and ensures that step sizes are not chosen too small (which might happen if we only use the sufficient decrease condition).

## Example

We use the [same example that we had when we explained the sufficient decrease condition](@ref sdc_id):

```@example cc
using SimpleSolvers # hide
using SimpleSolvers: CurvatureCondition, NewtonOptimizerCache, update!, gradient!, linesearch_objective, ldiv! # hide

x = [3., 1.3]
f = x -> 10 * sum(x .^ 3 / 6 - x .^ 2 / 2)
obj = MultivariateObjective(f, x)
hes = Hessian(obj, x; mode = :autodiff)
update!(hes, x)

c₂ = .9
g = gradient(x, obj)
rhs = -g
# the search direction is determined by multiplying the right hand side with the inverse of the Hessian from the left.
p = similar(rhs)
ldiv!(p, hes, rhs)
cc = CurvatureCondition(c₂, x, g, p, obj, obj.G)

# check different values
α₁, α₂, α₃, α₄, α₅ = .1, .4, 0.7, 1., 1.3
(cc(α₁), cc(α₂), cc(α₃), cc(α₄), cc(α₅))
```