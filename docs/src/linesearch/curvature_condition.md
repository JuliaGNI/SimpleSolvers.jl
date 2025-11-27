# The Curvature Condition

The *curvature condition* is used together with the [sufficient decrease condition](@ref "The Sufficient Decrease Condition") and ensures that step sizes are not chosen too small (which might happen if we only use the sufficient decrease condition). The sufficient decrease condition and the curvature condition together are called the *Wolfe conditions*.

## Standard Curvature Condition

For the standard curvature condition (see [`SimpleSolvers.CurvatureCondition`](@ref)) we have:

```math
    \frac{\partial}{\partial{}\alpha}\Bigg|_{\alpha=\alpha_k}f(R_{x_k}(\alpha{}p_k)) = g(\mathrm{grad}_{R_{x_k}(\alpha_k{}p_k)}f, p_k) \geq c_2g(\mathrm{grad}_{x_k}f, p_k) = c_2\frac{\partial}{\partial\alpha}\Bigg|_{\alpha=0}f(R_{x_k}(\alpha{}p_k)),
```

for ``c_2\in{}(c_1, 1).`` In words this means that the derivative with respect to ``\alpha_k`` should be bigger at the new iterate ``x_{k+1}`` than at the old iterate ``x_k``. 

## Strong Curvature Condition

For the strong curvature condition[^1] we *replace the curvature condition* by:

[^1]: We consequently also speak of the *strong Wolfe conditions* when taking the strong curvature condition and the sufficient decrease condition together.

```math
    |g(\mathrm{grad}_{R_{x_k}(\alpha_k{}p_k)}f, p_k)| < c_2|g(\mathrm{grad}_{x_k}f, p_k)|.
```

Note the sign change here. This is because the term ``g(\mathrm{grad}_{x_k}f, p_k)`` is negative if ``p_k`` is a [search direction](@ref "Line Search"). Both the standard *curvature condition* and the *strong curvature condition* are implemented under [`SimpleSolvers.CurvatureCondition`](@ref).

!!! info
    In order to use the corresponding condition you have to either pass `mode = :Standard` or `mode = :Strong` to the constructor of `CurvatureCondition`.


## Example

We use the [same example that we had when we explained the sufficient decrease condition](@ref sdc_example):

```@example cc
using SimpleSolvers # hide
using SimpleSolvers: CurvatureCondition, NewtonOptimizerCache, update!, gradient!, linesearch_problem, ldiv! # hide

x = [3., 1.3]
f = x -> 10 * sum(x .^ 3 / 6 - x .^ 2 / 2)
obj = OptimizerProblem(f, x)
hes = HessianAutodiff(obj, x)
update!(hes, x)

c₂ = .9
g = similar(x)
grad = GradientAutodiff{Float64}(obj.F, length(x))
gradient!(obj, grad, x)
rhs = -g
# the search direction is determined by multiplying the right hand side with the inverse of the Hessian from the left.
p = similar(rhs)
ldiv!(p, hes, rhs)
cc = CurvatureCondition(c₂, x, g, p, obj, grad)

# check different values
α₁, α₂, α₃, α₄, α₅ = .09, .4, 0.7, 1., 1.3
(cc(α₁), cc(α₂), cc(α₃), cc(α₄), cc(α₅))
```