# The Sufficient Decrease Condition

The *Armijo condition* or *sufficient decrease condition* states:

```math
    f(R_{x_k}(\alpha_k{}p_k)) \leq f(x_k) + c_1g_{x_k}(\alpha_k{}p_k, \mathrm{grad}^g_{x_k}f),  
```
for some constant ``c_1\in(0, 1)`` (see [`SimpleSolvers.DEFAULT_WOLFE_c₁`](@ref)).

The sufficient decrease condition can also be written as 

```math
    \frac{f(R_{x_k}(\alpha_k{}p_k)) - f(x_k)}{\alpha_k} \leq g_{x_k}(c_1p_k, \mathrm{grad}^g_{x_k}f).
```

As we assume that ``f(R_{x_k}(\alpha_k{}p_k)) \leq f(x_k)`` and ``g_{x_k}(c_1p_k, \mathrm{grad}^g_{x_k}f) < 0``, we can rewrite this as:

```math
    |\frac{f(R_{x_k}(\alpha_k{}p_k)) - f(x_k)}{\alpha_k}| \geq |g_{x_k}(c_1p_k, \mathrm{grad}^g_{x_k}f)|,
```

making clear why this is called the *sufficient decrease condition*. The parameter ``c_1`` is typically chosen very small, around ``10^{-4}``. This is implemented as [`SimpleSolvers.SufficientDecreaseCondition`](@ref).

## [Example](@id sdc_example_full)

We can visualize the sufficient decrease condition with an example:

```@example sdc
using SimpleSolvers # hide
using SimpleSolvers: SufficientDecreaseCondition, NewtonOptimizerCache, update!, linesearch_problem # hide

x = [3., 1.3]
f = x -> 10 * sum(x .^ 3 / 6 - x .^ 2 / 2)
obj = OptimizerProblem(f, x)
hes = HessianAutodiff(obj, x)
H = SimpleSolvers.alloc_h(x)
hes(H, x)

c₁ = 1e-4
grad = GradientAutodiff{Float64}(obj.F, length(x))
g = grad(x)
rhs = -g
# the search direction is determined by multiplying the right hand side with the inverse of the Hessian from the left.
p = similar(rhs)
p .= H \ rhs
sdc = SufficientDecreaseCondition(c₁, x, f(x), g, p, obj)

# check different values
α₁, α₂, α₃, α₄, α₅ = .09, .4, 0.7, 1., 1.3
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
arrows!(ax, [x[1]], [x[2]], [f(x)], [.15 * p[1]], [.15 * p[2]], [0.]; color=mred, linewidth=.01, arrowsize = .1, align=:tail)

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