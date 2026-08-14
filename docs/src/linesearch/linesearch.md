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

A line search is therefore a *sub-optimization problem* in an optimizer or solver in which we want to find an ``\alpha`` that minimizes:

```math
    \min_\alpha{}f^\mathrm{ls}(\alpha) = \min_\alpha{}f(\mathcal{R}_{x_k}(\alpha{}p_k)),
```
where ``p_k`` is the search direction and ``\mathcal{R}_{x_k}:T_{x_k}\mathcal{M}\to\mathcal{M}`` is a retraction at ``x_k.``

After having (i) found the search direction ``p_k``, (ii) defined the linesearch problem ``f^\mathrm{ls}`` and (iii) solved ``\alpha_k = \mathrm{argmin}_{\alpha}f(\alpha)`` we update ``x``:

```math
    x_{k+1} \gets \mathcal{R}_{x_k}(\alpha_k{}p_k).
```

In practice we will not be able to find the ideal ``\alpha`` at every step, but only an approximation thereof. Examples of line search algorithms that aim at finding this ``\alpha`` are the [static line search](@ref "Static Line Search"), the [backtracking line search](@ref "Backtracking Line Search") and the [strong Wolfe line search](@ref "Strong Wolfe Line Search").

## Line Search Problem

See the following docstrings:
- [`linesearch_problem`](@ref),
- [`LinesearchProblem`](@ref).

## Linesearches for Solvers

For solvers the output of ``f:\mathbb{R}^n\to\mathbb{R}^m`` is vector-valued. We therefore have
```math
f^\mathrm{ls}(\alpha) = ||f(x_k + \alpha{}p_k)||^2.
```

## Bounding the step

Every method returns ``\alpha \leq \alpha_\mathrm{max}``, and there are two ways to set that
ceiling — see [`SimpleSolvers.linesearch_αmax`](@ref).

The **method's own** ceiling is the `αmax` field of [`Bisection`](@ref), [`Quadratic`](@ref),
[`BierlaireQuadratic`](@ref) and [`StrongWolfe`](@ref), defaulting to
[`SimpleSolvers.DEFAULT_LINESEARCH_αmax`](@ref). It exists because a bracketing search grows its bracket outward
until the merit stops falling, so a nearly flat ``\varphi`` — or one whose minimiser is genuinely
far away — otherwise bounds the step only by the bracketing budget.

The **caller's** ceiling is an optional `αmax` field of the `params` passed to
[`solve`](@ref)/[`solve_with_status`](@ref):

```julia
solve_with_status(ls, one(T), (x = x, parameters = params, αmax = 2π / norm(direction)))
```

This one has to be per call because the scale it comes from is not a property of the merit. On the
manifold ``\mathcal{M}`` above, ``\alpha`` parameterizes a curve through a retraction, and past
about ``2\pi`` of rotation a longer step contributes nothing but round-off — a bound that is
geometric, and that changes with ``\|p_k\|`` at every step. Nothing the search itself can measure
sees it: ``f^\mathrm{ls}`` is *bounded* on a compact ``\mathcal{M}``, so ``f^\mathrm{ls}(10^9)``
can be genuinely lower than ``f^\mathrm{ls}(0)`` and every decrease test the search owns will
accept the step.

Supplying the ceiling as an input rather than clamping the returned ``\alpha`` is what keeps the
[`LinesearchStatus`](@ref) honest: the search stops at the ceiling, measures the merit there, and
reports the outcome of the step it actually hands back.
