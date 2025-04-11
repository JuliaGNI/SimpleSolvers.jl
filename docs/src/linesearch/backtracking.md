# Backtracking Line Search

A *backtracking line search method* determines the amount to move in a given search direction by iteratively decreasing a step size ``\alpha`` until an acceptable level is reached. In `SimpleSolvers` we can use the *Armijo* or the *Wolfe* conditions to quantify this *acceptable level*.

## Sufficient Decrease Condition

The *Armijo condition* or *sufficient decrease condition* states:

```math
    f(R_{x_k}(\alpha_k{}p_k)) \leq f(x_k) + c_1g_{x_k}(\alpha_k{}p_k, \mathrm{grad}^g_{x_k}f),  
```
for some constant ``c_1\in(0, 1)`` (see [`SimpleSolvers.DEFAULT_WOLFE_ϵ`](@ref)).

The sufficient decrease condition can also be written as 

```math
    \frac{f(R_{x_k}(\alpha_k{}p_k)) - f(x_k)}{\alpha_k} \leq g_{x_k}(c_1p_k, \mathrm{grad}^g_{x_k}f).
```

As we assume that ``f(R_{x_k}(\alpha_k{}p_k)) \leq f(x_k)`` and ``g_{x_k}(c_1p_k, \mathrm{grad}^g_{x_k}f) < 0``, we can rewrite this as:

```math
    |\frac{f(R_{x_k}(\alpha_k{}p_k)) - f(x_k)}{\alpha_k}| \geq |g_{x_k}(c_1p_k, \mathrm{grad}^g_{x_k}f)|,
```

making clear why this is called the *sufficient decrease condition*. The parameter ``c_1`` is typically chosen very small, around ``10^{-4}``. This is implemented as [`SimpleSolvers.SufficientDecreaseCondition`](@ref).

We can visualize the sufficient decrease condition with an example:

```@example sdc
using SimpleSolvers # hide
using SimpleSolvers: SufficientDecreaseCondition, NewtonOptimizerCache, update!, gradient!, linesearch_objective, ldiv! # hide

x = [3., 1.3]
f = x -> 10 * sum(x .^ 3 / 6 - x .^ 2 / 2)
obj = MultivariateObjective(f, x)
hes = Hessian(obj, x; mode = :autodiff)
update!(hes, x)

c₁ = 1e-4
g = gradient(x, obj)
rhs = -g
# the search direction is determined by multiplying the right hand side with the inverse of the Hessian from the left.
p = similar(rhs)
ldiv!(p, hes, rhs)
sdc = SufficientDecreaseCondition(c₁, x, f(x), g, p, obj)

# check different values
α₁, α₂, α₃, α₄, α₅ = .1, .4, 0.7, 1., 1.3
(sdc(α₁), sdc(α₂), sdc(α₃), sdc(α₄), sdc(α₅))
```

We can also illustrate this:

```@setup sdc
using CairoMakie, LaTeXStrings
mred = RGBf(214 / 256, 39 / 256, 40 / 256)
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256)
mgreen = RGBf(44 / 256, 160 / 256, 44 / 256)
mblue = RGBf(31 / 256, 119 / 256, 180 / 256)
morange = RGBf(255 / 256, 127 / 256, 14 / 256)

fig = Figure()
ax = Axis3(fig[1,1])
xs = LinRange(-.2, 3.2, 100)
ys = LinRange(1., 4., 100)
zs = [f(vcat(x, y)) for x in xs, y in ys]
surface!(ax, xs, ys, zs; alpha = .5)
scatter!(ax, [x[1]], [x[2]], [f(x)]; color=mred, label=L"x_0")
arrows!(ax, [x[1]], [x[2]], [f(x)], [.15 * p[1]], [.15 * p[2]], [0.]; color=mred, linewidth=.01, arrowsize = .1, align=:origin)

x1 = x + α₁ * p
x2 = x + α₂ * p
x3 = x + α₃ * p
x4 = x + α₄ * p
x5 = x + α₅ * p
scatter!(ax, [x1[1]], [x1[2]], [f(x1)]; color=mpurple, label=L"x_1")
scatter!(ax, [x2[1]], [x2[2]], [f(x2)]; color=morange, label=L"x_2")
scatter!(ax, [x3[1]], [x3[2]], [f(x3)]; color=mblue, label=L"x_3")
scatter!(ax, [x4[1]], [x4[2]], [f(x4)]; color=mgreen, label=L"x_4")
scatter!(ax, [x5[1]], [x5[2]], [f(x5)]; color=mred, label=L"x_5")

axislegend(ax)
save("sufficient_decrease.png", fig)
nothing
```

![](sufficient_decrease.png)

## Wolfe Conditions

For the Wolfe conditions we have, in addition to the sufficient decrease condition, another condition called the *curvature condition*:

```math
    \frac{\partial}{\partial{}\alpha_k}f(R_{x_k}(\alpha_k{}p_k)) = g(\mathrm{grad}_{R_{x_k}(\alpha_k{}p_k)}f, p_k) \geq c_2g(\mathrm{grad}_{x_k}f, p_k) = c_2\frac{\partial}{\partial\alpha}\Bigg|_{\alpha=0}f(R_{x_k}(\alpha{}p_k)),
```

for ``c_2\in{}(c_1, 1).`` In words this means that the derivative with respect to ``\alpha_k`` should be bigger at the new iterate ``x_{k+1}`` than at the old iterate ``x_k``. 

## Strong Wolfe Conditions

For the strong Wolfe conditions we *replace the curvature condition* by:

```math
    |g(\mathrm{grad}_{R_{x_k}(\alpha_k{}p_k)}f, p_k)| < c_2|g(\mathrm{grad}_{x_k}f, p_k)|.
```

Note the sign change here. This is because the term ``g(\mathrm{grad}_{x_k}f, p_k)`` is negative. Both the standard *curvature condition* and the *strong curvature condition* are implemented under [`SimpleSolvers.CurvatureCondition`](@ref).


## Example

We show how to use linesearches in `SimpleSolvers` to solve a simple toy problem[^1]:

[^1]: Also compare this to the case of the [static line search](@ref static_example).

```@example sdc
using SimpleSolvers # hide

sl = Backtracking()
nothing # hide
```

`SimpleSolvers` contains a function [`SimpleSolvers.linesearch_objective`](@ref) that allocates a [`SimpleSolvers.TemporaryUnivariateObjective`](@ref) that only depends on ``\alpha``:

```@example sdc
using SimpleSolvers: linesearch_objective, NewtonOptimizerCache, LinesearchState, update! # hide
cache = NewtonOptimizerCache(x)

gradient!(obj, x)
update!(cache, x, obj.g, hes)
ls_obj = linesearch_objective(obj, cache)
nothing # hide
```

We now use this to compute a *backtracking line search*[^2]:

[^2]: We also note the use of the [`SimpleSolvers.LinesearchState`](@ref) constructor here, which has to be used together with a [`SimpleSolvers.LinesearchMethod`](@ref).

```@example sdc
ls = LinesearchState(sl)
α = 50.
αₜ = ls(ls_obj, α)
```

And we check whether the [`SimpleSolvers.SufficientDecreaseCondition`](@ref) is satisfied:
```@example sdc
sdc(αₜ)
```

!!! info
    We note that for the static line search we always just return ``\alpha``.

## Backtracking Line Search for a Line Search Objective

We note that when performing backtracking on a [line search objective](@ref "Line Search Objective") care needs to be taken. This is because we need to find equivalent quantities for ``\grad_{x_k}f`` and ``p``. We first look at the derivative of the line search objective:

```math
\frac{d}{d\alpha}f^\mathrm{ls}(\alpha) = \frac{d}{d\alpha}f(\mathcal{R}_{x_k}(\alpha{}p)) = \langle d|_{\mathcal{R}_{x_k}(\alpha{}p)}f, \alpha{}p \rangle,
```
because the tangent map of a retraction is the identity at zero, i.e. ``T_{0_x}\mathcal{R} = \mathrm{id}_{T_x\mathcal{M}}.