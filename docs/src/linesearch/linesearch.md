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

A line search is therefore a *sub-optimization problem* in a nonlinear [optimizer](@ref "Optimizers") (or solver) in which we want to find an ``\alpha`` that minimizes:

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

## Linesearches for Optimizers

In `SimpleSolvers` we typically build the search direction by multiplying the gradient with a [Hessian](@ref "Hessians"). When starting at ``x_k`` we take:

```math
    p_k = H_{x_k}^{-1}(\nabla_{x_k}f),
```
where ``[H_{x_k}]_{ij} = \partial^2{}f\partial{}x_i\partial{}x_j|_{x_k}`` is the [Hessian](@ref "Hessians"). Note that we often use approximations of this Hessian in practice (such as the [`HessianBFGS`](@ref)).

The linesearch objective is then built as
```math
f^\mathrm{ls}(\alpha) = f(x_k + \alpha{}p_k).
```

For manifolds [absil2008optimization](@cite) defining a Hessian, equivalently to defining a [gradient](@ref "Gradients"), requires a Riemannian metric and the associated Levi-Civita connection ``\nabla``:

```math
\mathrm{Hess}(f) := \nabla\nabla{}f = \nabla{}df \in \Gamma(T^*\mathcal{M}\otimes{}T^*\mathcal{M}).
```

For specific vector fields ``\xi, \eta \in \Gamma(T\mathcal{M})`` we can write this as:

```math
\langle \mathrm{Hess}(f)[\xi], \eta  \rangle = \xi(\eta{}f) - (\nabla_\xi\eta)f.
```