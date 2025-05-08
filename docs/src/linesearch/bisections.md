# Bisections

[`Bisection`](@ref)s work by moving an interval until we observe one in which the sign of the derivative of the function changes. 

## Example

We consider the same example as we had when demonstrating [backtracking line search](@ref "Backtracking Line Search for a Line Search Objective"):

```@setup bisection
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

```@example bisection
ls_obj = linesearch_objective(obj, cache)
nothing # hide
```

## Bracketing

For bracketing [kochenderfer2019algorithms](@cite) we move an interval successively and simultaneously increase it in the hope that we observe a local minimum (see [`bracket_minimum`](@ref)).

```@example bisection
α₀ = 0.0
(a, c) = bracket_minimum(Function(ls_obj), α₀)
```

```@setup bisection
alpha = 0.:.01:1.5

y = ls_obj.(alpha)
fig = Figure()
ax = Axis(fig[1, 1]; xlabel = L"\alpha", ylabel = L"f^\mathrm{ls}(\alpha)")
lines!(ax, alpha, y)

scatter!(ax, [α₀], [ls_obj(α₀)]; color=mred, label=L"\alpha_0")
vlines!(ax, [a]; label = L"a", color=mpurple)
vlines!(ax, [c]; label = L"c", color=mgreen)

axislegend(ax)
save("2d_plot.png", fig)
nothing
```

![](2d_plot.png)

We then use this interval to start the bisection algorithm.

### Potential Problem with Backtracking

We here illustrate a potential issue with backtracking. For this consider the following function:

```@example bisection
using SimpleSolvers: bracket_root
f2(α::T) where {T <: Number} = α^2 - one(T)
α₀ = -3.0
(a, c) = bracket_root(f2, α₀)
```

And when we plot this we find:

```@setup bisection
alpha = -3.5:.01:2.5

y = f2.(alpha)
fig = Figure()
ax = Axis(fig[1, 1]; xlabel = L"\alpha", ylabel = L"f^_2(\alpha)")
lines!(ax, alpha, y)

scatter!(ax, [α₀], [f2(α₀)]; color=mred, label=L"\alpha_0")
scatter!(ax, [-1.], [0.]; label=L"r_1", color=morange)
scatter!(ax, [1.], [0.]; label=L"r_2", color=mblue)
vlines!(ax, [a]; label = L"a", color=mpurple)
vlines!(ax, [c]; label = L"c", color=mgreen)

axislegend(ax)
save("2d_plot_issue.png", fig)
nothing
```

![](2d_plot_issue.png)

And we see that the interval now contains two roots, ``r_1`` and ``r_2``.