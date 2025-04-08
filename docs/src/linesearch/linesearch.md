# Line Search 

This page is largely a summary of [nocedal2006numerical; Chapter 3](@cite). We summarize this reference by omitting proofs, but also aim to extend it to manifolds.

A line search method has the goal of minimizing an objective (either a [`UnivariateObjective`](@ref) or a [`MultivariateObjective`](@ref)) approximately, based on a *search direction*[^1].
[^1]: in [nocedal2006numerical](@cite) a *search direction* is called a **descent direction**.

!!! info "Definition"
    For an objective ``f:\mathcal{M}\to\mathbb{R}`` on a manifold ``\mathcal{M}`` a **search direction** at point ``x_k\in\mathcal{M}`` is a vector ``p_k\in{}T_{x_k}\mathcal{M}`` with
    ```math
        g_{x_k}(p_k, \mathrm{grad}^g_{x_k}f) < 0,
    ```
    where ``g_{x_k}:T_{x_k}\mathcal{M}\times{}T_{x_k}\mathcal{M}\to\mathbb{R}`` is a Riemannian metric.

A line search is therefore a *sub-optimization problem* in a nonlinear optimizer (or solver) in which we want to find an ``\alpha`` that minimizes:

```math
    \min_\alpha{}f(\alpha) = \min_\alpha{}F(x_k + \alpha{}p_k),
```
where ``p_k`` is the search direction.

For line search methods we have to (i) find a search direction ``p_k`` and (ii) find an appropriate step size ``\alpha_k``. We then update ``x_k`` based on these quantities:

```math
    x_{k+1} \gets R_{x_k}(\alpha_k{}p_k),
```
where ``R_{x_k}:T_{x_k}\mathcal{M}\to\mathcal{M}`` is a retraction at ``x_k.``