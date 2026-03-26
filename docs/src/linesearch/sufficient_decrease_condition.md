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

making clear why this is called the *sufficient decrease condition*. The parameter ``c_1`` is typically chosen very small, around ``10^{-4}``. This is implemented as [`SufficientDecreaseCondition`](@ref).

## [Example](@id sdc_example_full)

We include an example:

```@example sdc
using SimpleSolvers # hide
using SimpleSolvers: SufficientDecreaseCondition, update!, linesearch_problem, NullParameters, direction # hide
using SimpleSolvers: direction!, cache # hide

x = [3., 1.3]
y = similar(x)
f(y, x, params) = y .= 10 .* x .^ 3 ./ 6 .- x .^ 2 ./ 2
_params = NullParameters()
f(y, x, _params)
s = NewtonSolver(x, y; F = f)
c₁ = 1e-4
state = NonlinearSolverState(x)
update!(state, x, y)
direction!(s, x, _params, 0)
p = copy(direction(cache(s))) # hide
ls_obj = linesearch_problem(s)
params = (x = state.x, parameters = _params)
sdc = SufficientDecreaseCondition(c₁, ls_obj.F(0., params), ls_obj.D(0., params), alpha -> ls_obj.F(alpha, params))

# check different values
α₁, α₂, α₃, α₄, α₅ = .09, .4, 0.7, 1., 1.3
(sdc(α₁), sdc(α₂), sdc(α₃), sdc(α₄), sdc(α₅))
```

We further illustrate this:

```@setup sdc
using SimpleSolvers: l2norm
using CairoMakie, LaTeXStrings
mred = RGBf(214 / 256, 39 / 256, 40 / 256)
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256)
mgreen = RGBf(44 / 256, 160 / 256, 44 / 256)
mblue = RGBf(31 / 256, 119 / 256, 180 / 256)
morange = RGBf(255 / 256, 127 / 256, 14 / 256)

fig = Figure()
ax = Axis3(fig[1,1])
xs = LinRange(-.2, 3.2, 100)
ys = LinRange(.5, 1.5, 100)
_y = similar(x)
zs = [l2norm(f(_y, vcat(x, y), _params)) for x in xs, y in ys]
surface!(ax, xs, ys, zs; alpha = .5)
scatter!(ax, [x[1]], [x[2]], [l2norm(f(_y, x, _params))]; color=mred, label=L"x_0")
arrows!(ax, [x[1]], [x[2]], [l2norm(f(_y, x, _params))], [.15 * p[1]], [.15 * p[2]], [0.]; color=mred, linewidth=.01, arrowsize = .1, align=:tail)

x1 = x + α₁ * p
x2 = x + α₂ * p
x3 = x + α₃ * p
x4 = x + α₄ * p
x5 = x + α₅ * p
scatter!(ax, [x1[1]], [x1[2]], [l2norm(f(y, x1, _params))]; color=mpurple, label=L"x_1")
scatter!(ax, [x2[1]], [x2[2]], [l2norm(f(y, x2, _params))]; color=morange, label=L"x_2")
scatter!(ax, [x3[1]], [x3[2]], [l2norm(f(y, x3, _params))]; color=mblue, label=L"x_3")
scatter!(ax, [x4[1]], [x4[2]], [l2norm(f(y, x4, _params))]; color=mgreen, label=L"x_4")
scatter!(ax, [x5[1]], [x5[2]], [l2norm(f(y, x5, _params))]; color=mred, label=L"x_5")

axislegend(ax)
save("sufficient_decrease_light.png", fig)
save("sufficient_decrease_dark.png", fig)
nothing
```

![Example of points that largely satisfy the *sufficient decrease condition*.](sufficient_decrease_dark.png)
![Example of points that largely satisfy the *sufficient decrease condition*.](sufficient_decrease_light.png)
