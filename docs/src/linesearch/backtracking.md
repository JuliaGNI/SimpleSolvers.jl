# Backtracking Line Search

A *backtracking line search method* determines the amount to move in a given search direction by iteratively decreasing a step size ``\alpha`` until an acceptable level is reached. In `SimpleSolvers` we can use the [sufficient decrease condition](@ref "The Sufficient Decrease Condition") and the [curvature condition](@ref "The Curvature Condition") to quantify this *acceptable level*. The sufficient decrease condition is also referred to as the *Armijo condition* and the sufficient decrease condition and the curvature condition are referred to as the *Wolfe conditions*[^1] [nocedal2006numerical](@cite). 

[^1]: If we use the [strong curvature condition](@ref "Strong Curvature Condition") instead of the [standard curvature condition](@ref "Standard Curvature Condition") we conversely also say that we use the *strong Wolfe conditions*.

!!! info
    We note that for the static line search we always just return ``\alpha``.


## Backtracking Line Search for a Line Search Objective

We note that when performing backtracking on a [line search objective](@ref "Line Search Objective") care needs to be taken. This is because we need to find equivalent quantities for ``\mathrm{grad}_{x_k}f`` and ``p``. We first look at the derivative of the line search objective:

```math
\frac{d}{d\alpha}f^\mathrm{ls}(\alpha) = \frac{d}{d\alpha}f(\mathcal{R}_{x_k}(\alpha{}p)) = \langle d|_{\mathcal{R}_{x_k}(\alpha{}p)}f, \alpha{}p \rangle,
```
because the tangent map of a retraction is the identity at zero [absil2008optimization](@cite), i.e. ``T_{0_x}\mathcal{R} = \mathrm{id}_{T_x\mathcal{M}}``. In the equation above ``d|_{\mathcal{R}_{x_k}(\alpha{}p)}f\in{}T^*\mathcal{M}`` indicates the exterior derivative of ``f`` evaluated at ``\mathcal{R}_{x_k}(\alpha{}p)`` and ``\langle \cdot, \cdot \rangle: T^*\mathcal{M}\times{}T\mathcal{M}\to\mathbb{R}`` is the natural pairing between tangent and cotangent space[^2] [bishop1980tensor](@cite).

[^2]: If we are not dealing with general Riemannian manifolds but only vector spaces then ``d|_{\mathcal{R}_{x_k}(\alpha{}p)}f`` simply becomes ``\nabla_{\mathcal{R}_{x_k}(\alpha{}p)}f`` and we further have ``\langle A, B\rangle = A^T B``.

We again look at [the example introduced when talking about the sufficient decrease condition](@ref sdc_example_full) and cast it in the form of a *line search objective*:

```@setup ls_obj
using SimpleSolvers # hide
using SimpleSolvers: SufficientDecreaseCondition, NewtonOptimizerCache, update!, gradient!, linesearch_objective, ldiv! # hide

x = [3., 1.3]
f = x -> 10 * sum(x .^ 3 / 6 - x .^ 2 / 2)
obj = MultivariateObjective(f, x)
value!(obj, x)
hes = Hessian(obj, x; mode = :autodiff)
update!(hes, x)

c₁ = 1e-4
g = gradient!(obj, x)
rhs = -g
# the search direction is determined by multiplying the right hand side with the inverse of the Hessian from the left.
p = similar(rhs)
ldiv!(p, hes, rhs)
sdc = SufficientDecreaseCondition(c₁, x, f(x), g, p, obj)

# check different values
α₁, α₂, α₃, α₄, α₅ = .09, .4, 0.7, 1., 1.3

using CairoMakie, LaTeXStrings
mred = RGBf(214 / 256, 39 / 256, 40 / 256)
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256)
mgreen = RGBf(44 / 256, 160 / 256, 44 / 256)
mblue = RGBf(31 / 256, 119 / 256, 180 / 256)
morange = RGBf(255 / 256, 127 / 256, 14 / 256)

using SimpleSolvers: linesearch_objective, NewtonOptimizerCache, LinesearchState, update! # hide
cache = NewtonOptimizerCache(x)
update!(cache, x, obj.g, hes)
nothing # hide
```

```@example ls_obj
ls_obj = linesearch_objective(obj, cache)
nothing # hide
```

This objective only depends on the parameter ``\alpha``. We plot it:

```@setup ls_obj
alpha = 0.:.01:1.5

y = ls_obj.(alpha)
fig = Figure()
ax = Axis(fig[1, 1]; xlabel = L"\alpha", ylabel = L"f^\mathrm{ls}(\alpha)")
lines!(ax, alpha, y)

scatter!(ax, [α₁], [ls_obj(α₁)]; color=mpurple, label=L"\alpha_1")
scatter!(ax, [α₂], [ls_obj(α₂)]; color=morange, label=L"\alpha_2")
scatter!(ax, [α₃], [ls_obj(α₃)]; color=mblue, label=L"\alpha_3")
scatter!(ax, [α₄], [ls_obj(α₄)]; color=mgreen, label=L"\alpha_4")
scatter!(ax, [α₅], [ls_obj(α₅)]; color=mred, label=L"\alpha_5")

axislegend(ax)

save("ls_backtracking_2d_plot.png", fig)
nothing
```

![](ls_backtracking_2d_plot.png)

## [Example](@id sdc_example)

We show how to use linesearches in `SimpleSolvers` to solve a simple toy problem[^3]:

[^3]: Also compare this to the case of the [static line search](@ref static_example).

```@example ls_obj
using SimpleSolvers # hide

sl = Backtracking()
nothing # hide
```

`SimpleSolvers` contains a function [`SimpleSolvers.linesearch_objective`](@ref) that allocates a [`SimpleSolvers.TemporaryUnivariateObjective`](@ref) that only depends on ``\alpha``:

We now use this to compute a *backtracking line search*[^4]:

[^4]: We also note the use of the [`SimpleSolvers.LinesearchState`](@ref) constructor here, which has to be used together with a [`SimpleSolvers.LinesearchMethod`](@ref).

```@example ls_obj
ls = LinesearchState(sl)
α = 50.
αₜ = ls(ls_obj, α)
```

And we check whether the [`SimpleSolvers.SufficientDecreaseCondition`](@ref) is satisfied:
```@example ls_obj
sdc = SufficientDecreaseCondition(c₁, x, f(x), g, p, obj)
sdc(αₜ)
```

Similarly for the [`SimpleSolvers.CurvatureCondition`](@ref):

```@example ls_obj
using SimpleSolvers: CurvatureCondition # hide
c₂ = .9
cc = CurvatureCondition(c₂, x, g, p, obj, obj.G)
cc(αₜ)
```