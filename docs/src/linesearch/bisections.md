# Bisections

[`Bisection`](@ref)s work by moving an interval until we observe a sign change (either in the function or its derivative). 

## Example

We consider the same example as we had when demonstrating [backtracking line search](@ref "Backtracking Line Search for a Line Search Problem"):

```@setup bisection
using SimpleSolvers # hide
using SimpleSolvers: SufficientDecreaseCondition, update!, linesearch_problem, NullParameters, direction # hide
using SimpleSolvers: direction!, cache # hide

x = [3., 1.3]
f = x -> 10 * sum(x .^ 3 / 6 - x .^ 2 / 2)
obj = OptimizerProblem(f, x)
hes = HessianAutodiff(obj, x)

c₁ = 1e-4
grad = GradientAutodiff{Float64}(obj.F, length(x))
# the search direction is determined by multiplying the right hand side with the inverse of the Hessian from the left.
state = NewtonOptimizerState(x)
cache = NewtonOptimizerCache(x)
problem = linesearch_problem(obj, grad, cache)
update!(state, grad, x)
update!(cache, state, grad, hes, x)

params = (x = state.x,)
sdc = SufficientDecreaseCondition(c₁, problem.F(0., params), problem.D(0., params), alpha -> problem.F(alpha, params))
# check different values
α₁, α₂, α₃, α₄, α₅ = .09, .4, 0.7, 1., 1.3

using CairoMakie, LaTeXStrings
mred = RGBf(214 / 256, 39 / 256, 40 / 256)
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256)
mgreen = RGBf(44 / 256, 160 / 256, 44 / 256)
mblue = RGBf(31 / 256, 119 / 256, 180 / 256)
morange = RGBf(255 / 256, 127 / 256, 14 / 256)
```

```@example bisection
ls_obj = linesearch_problem(s)
nothing # hide
```

## Bracketing

Performing bisections requires providing an *initial interval*. If we are only given a single point instead of an interval we have to perform *bracketing*.
For bracketing [kochenderfer2019algorithms](@cite) we move an interval successively and simultaneously increase it in the hope that we observe a local minimum (see [`bracket_minimum`](@ref)).

```@example bisection
α₀ = 0.0
(a, c) = bracket_minimum(alpha -> ls_obj.F(alpha, params), α₀)
```

```@setup bisection
alpha = 0.:.01:1.5

y = [ls_obj.F(_alpha, params) for _alpha in alpha]
fig = Figure()
ax = Axis(fig[1, 1]; xlabel = L"\alpha", ylabel = L"f^\mathrm{ls}(\alpha)")
lines!(ax, alpha, y)

scatter!(ax, [α₀], [ls_obj.F(α₀, params)]; color=mred, label=L"\alpha_0")
vlines!(ax, [a]; label = L"a", color=mpurple)
vlines!(ax, [c]; label = L"c", color=mgreen)

axislegend(ax)
save("2d_plot_dark.png", fig)
save("2d_plot_light.png", fig)
nothing
```

![](2d_plot_dark.png)
![](2d_plot_light.png)

We then use this interval to start the bisection algorithm.

### Potential Problem with Backtracking

We here illustrate a potential issue with backtracking. For this consider the following function:

```@example bisection
using SimpleSolvers: bracket_root # hide
f2(α::T) where {T <: Number} = α^2 - one(T)
α₀ = -10.0
(a, c) = bracket_root(f2, α₀)
```

And when we plot this we find:

```@setup bisection
alpha = -(-α₀ + .5):.01:2.5

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
save("2d_plot_issue_light.png", fig)
save("2d_plot_issue_dark.png", fig)
nothing
```

![](2d_plot_issue_light.png)
![](2d_plot_issue_dark.png)

If the interval would contain ``r_1`` and ``r_2`` then we get an error:

```@example bisection
struct UnexpectedSuccess <: Exception end #hide
try  #hide
bracket_root(f2, 30.)
throw(UnexpectedSuccess()) #hide
catch e; e isa UnexpectedSuccess ? rethrow(e) : showerror(stderr, e); end  #hide
```
