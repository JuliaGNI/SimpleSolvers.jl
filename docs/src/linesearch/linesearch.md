# Linesearch 

This page is a summary of [nocedal2006numerical; Chapter 3](@cite).

A linesearch method has the goal of minimizing an objective (either a [`UnivariateObjective`](@ref) or a [`MultivariateObjective`](@ref)) approximately, based on a *search direction*.

!!! info "Definition"
    A **search direction for a univariate objective** is the derivative of the objective, i.e.
    ```math
        f'(\alpha).
    ```
    A **search direction for a multivariate objective** is the gradient of the objective, i.e.
    ```math
        \nabla_\alpha{}f.
    ```

