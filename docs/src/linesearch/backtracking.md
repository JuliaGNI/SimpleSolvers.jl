# Backtracking Line Search

A *backtracking line search method* determines the amount to move in a given search direction by iteratively decreasing a step size ``\alpha`` until an acceptable level is reached. In `SimpleSolvers` we can use the *Armijo* or the *Wolfe* conditions to quantify this *acceptable level*.

## Armijo Condition

The Armijo condition is the following:

```math
    \frac{f(\alpha) - f(\alpha_0)}{\epsilon} < \alpha.
```

### Sufficient Decrease Condition

```math
    f(R_{x_k}(\alpha_k{}p_k)) < f(x_k) + c_1g_{x_k}(\alpha_k{}p_k, \mathrm{grad}^g_{x_k}),  
```
for some constant ``c_1\in(0, 1)`` (see [`SimpleSolvers.DEFAULT_WOLFE_ϵ`](@ref)).

The sufficient decrease condition can also be written as 

```math
    \frac{f(R_{x_k}(\alpha_k{}p_k)) - f(x_k)}{\alpha_k} < g_{x_k}(c_1p_k, \mathrm{grad}^g_{x_k}).
```

The parameter ``c_1`` is typically chosen very small, around ``10^{-4}``. This is implemented as [`SufficientDecreaseCondition`](@ref).

## Wolfe Conditions

For the Wolfe conditions we have, in addition to the sufficient decrease condition, another condition called the *curvature condition*:

```math
    \frac{\partial}{\partial{}\alpha_k}f(R_{x_k}(\alpha_k{}p_k)) = g(\mathrm{grad}_{R_{x_k}(\alpha_k{}p_k)}f, p_k) \geq c_2g(\mathrm{grad}_{x_k}f, p_k) = c_2\frac{\partial}{\partial\alpha}\Bigg|_{\alpha=0}f(R_{x_k}(\alpha{}p_k)),
```

for ``c_2\in{}(c_1, 1).``

## Strong Wolfe Conditions

For the strong Wolfe conditions we *replace the curvature condition* by:

```math
    |g(\mathrm{grad}_{R_{x_k}(\alpha_k{}p_k)}f, p_k)| < c_2|g(\mathrm{grad}_{x_k}f, p_k)|.
```

Note the sign change here. This is because the term ``g(\mathrm{grad}_{x_k}f, p_k)`` is negative.