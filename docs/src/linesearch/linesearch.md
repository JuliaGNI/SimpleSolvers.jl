# Line Search 

This page is largely a summary of [nocedal2006numerical; Chapter 3](@cite). We further extend some of the methodology contained in this reference to manifolds.

A line search method has the goal of minimizing a [`LinesearchProblem`](@ref) approximately, based on a *search direction*[^1].

[^1]: in [nocedal2006numerical](@cite) (and other references) a *search direction* is called a *descent direction*.

!!! info "Definition"
    For an optimizer problem ``f:\mathcal{M}\to\mathbb{R}`` on a manifold ``\mathcal{M}`` a **search direction** at point ``x_k\in\mathcal{M}`` is a vector ``p_k\in{}T_{x_k}\mathcal{M}`` for which we have
    ```math
        g_{x_k}(p_k, \mathrm{grad}^g_{x_k}f) < 0,
    ```
    where ``g_{x_k}:T_{x_k}\mathcal{M}\times{}T_{x_k}\mathcal{M}\to\mathbb{R}`` is a Riemannian metric.

A line search is therefore a *sub-optimization problem* in an optimizer (or solver) in which we want to find an ``\alpha`` that minimizes:

```math
    \min_\alpha{}f^\mathrm{ls}(\alpha) = \min_\alpha{}f(\mathcal{R}_{x_k}(\alpha{}p_k)),
```
where ``p_k`` is the search direction and ``\mathcal{R}_{x_k}:T_{x_k}\mathcal{M}\to\mathcal{M}`` is a retraction at ``x_k.``

After having (i) found the search direction ``p_k``, (ii) defined the linesearch problem ``f^\mathrm{ls}`` and (iii) solved ``\alpha_k = \mathrm{argmin}_{\alpha}f(\alpha)`` we update ``x``:

```math
    x_{k+1} \gets \mathcal{R}_{x_k}(\alpha_k{}p_k).
```

In practice we will not be able to find the ideal ``\alpha`` at every step, but only an approximation thereof. Examples of line search algorithms that aim at finding this ``\alpha`` are the [static line search](@ref "Static Line Search") and the [backtracking line search](@ref "Backtracking Line Search").

## Line Search Problem

See the following docstrings:
- [`linesearch_problem`](@ref),
- [`LinesearchProblem`](@ref).

## Linesearches for Solvers

For solvers the output of ``f:\mathbb{R}^n\to\mathbb{R}^m`` is vector-valued. We therefore have
```math
f^\mathrm{ls}(\alpha) = ||f(x_k + \alpha{}p_k)||^2.
```
