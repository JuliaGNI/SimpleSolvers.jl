# Backtracking Line Search

A *backtracking line search method* determines the amount to move in a given search direction by iteratively decreasing a step size ``\alpha`` until an acceptable level is reached. In `SimpleSolvers` we can use the *Armijo* or the *Wolfe* conditions to quantify this *acceptable level*.

## Armijo condition

The Armijo condition is the following:

```math
    \frac{f(\alpha) - f(\alpha_0)}{\epsilon} < \alpha.
```

## Wolfe Condition

The Wolfe condition is the following:

```math
    \frac{f(\alpha) - f(\alpha_0)}{\epsilon} < \alpha\cdot{}f'(\alpha_0).
```