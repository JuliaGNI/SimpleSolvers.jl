# Strong Wolfe Line Search

The [`StrongWolfe`](@ref) line search finds a step length ``\alpha`` that satisfies the **strong Wolfe conditions**,

```math
\begin{aligned}
f^\mathrm{ls}(\alpha) &\leq f^\mathrm{ls}(0) + c_1\,\alpha\,{f^\mathrm{ls}}'(0), &\text{(sufficient decrease / Armijo)}\\
|{f^\mathrm{ls}}'(\alpha)| &\leq c_2\,|{f^\mathrm{ls}}'(0)|, &\text{(strong curvature)}
\end{aligned}
```

with ``0 < c_1 < c_2 < 1``. Unlike the [backtracking line search](@ref "Backtracking Line Search"), which enforces only the sufficient-decrease (Armijo) condition — the curvature condition cannot be honoured by shrinking ``\alpha`` alone — `StrongWolfe` genuinely enforces the curvature condition. It does so with the bracketing line search of [nocedal2006numerical; Alg. 3.5 and 3.6 (`zoom`)](@cite):

1. a **bracketing** phase grows ``\alpha`` (doubling, up to [`SimpleSolvers.DEFAULT_WOLFE_αmax`](@ref)) until an interval containing a point that satisfies the conditions is found, then
2. a **zoom** phase shrinks that interval by bisection until the strong Wolfe conditions hold.

Enforcing the curvature condition requires the derivative ``{f^\mathrm{ls}}'`` at each trial step, so `StrongWolfe` is more expensive than [`Backtracking`](@ref). Use it when curvature control is genuinely required.

!!! info
    The strong Wolfe conditions require a *descent direction*, i.e. ``{f^\mathrm{ls}}'(0) < 0``. If the line search problem is not decreasing at ``\alpha = 0`` the method cannot make progress; it returns the caller's initial step and (at `verbosity ≥ 1`) warns.

See the docstring of [`StrongWolfe`](@ref) for the available keywords.
