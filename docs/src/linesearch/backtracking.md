# Backtracking Line Search

A *backtracking line search method* determines the amount to move in a given search direction by iteratively decreasing a step size ``\alpha`` until an acceptable level is reached. In `SimpleSolvers` we can use the [sufficient decrease condition](@ref "The Sufficient Decrease Condition") and the [curvature condition](@ref "The Curvature Condition") to quantify this *acceptable level*. The sufficient decrease condition is also referred to as the *Armijo condition* and the sufficient decrease condition and the curvature condition are referred to as the *Wolfe conditions*[^1] [nocedal2006numerical](@cite). 

[^1]: If we use the [strong curvature condition](@ref "Strong Curvature Condition") instead of the [standard curvature condition](@ref "Standard Curvature Condition") we conversely also say that we use the *strong Wolfe conditions*.

## Backtracking Line Search for a Line Search Problem

We note that the Wolfe conditions can be written very concisely by using [line search problems](@ref "Line Search Problem"):

```math
\frac{d}{d\alpha}f^\mathrm{ls}(\alpha) = \frac{d}{d\alpha}f(\mathcal{R}_{x_k}(\alpha{}p)) = \langle d|_{\mathcal{R}_{x_k}(\alpha{}p)}f, \alpha{}p \rangle,
```
where the tangent map of a retraction is the identity at zero [absil2008optimization](@cite), i.e. ``T_{0_x}\mathcal{R} = \mathrm{id}_{T_x\mathcal{M}}``. In the equation above ``d|_{\mathcal{R}_{x_k}(\alpha{}p)}f\in{}T^*\mathcal{M}`` indicates the exterior derivative of ``f`` evaluated at ``\mathcal{R}_{x_k}(\alpha{}p)`` and ``\langle \cdot, \cdot \rangle: T^*\mathcal{M}\times{}T\mathcal{M}\to\mathbb{R}`` is the natural pairing between tangent and cotangent space[^2] [bishop1980tensor](@cite).

[^2]: If we are not dealing with general Riemannian manifolds but only vector spaces then ``d|_{\mathcal{R}_{x_k}(\alpha{}p)}f`` simply becomes ``\nabla_{\mathcal{R}_{x_k}(\alpha{}p)}f`` and we further have ``\langle A, B\rangle = A^T B``.

We again look at [the example introduced when talking about the sufficient decrease condition](@ref sdc_example_full) and cast it in the form of a *line search problem*:

```@setup ls_obj
using SimpleSolvers # hide
using SimpleSolvers: SufficientDecreaseCondition, NewtonOptimizerCache, update!, linesearch_problem, ldiv!, direction # hide

x = [3., 1.3]
f = x -> 10 * sum(x .^ 3 / 6 - x .^ 2 / 2)
obj = OptimizerProblem(f, x)
hes = HessianAutodiff(obj, x)

c₁ = 1e-4
grad = GradientAutodiff{Float64}(obj.F, length(x))
g = grad(x)
rhs = -g
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

update!(cache, state, grad, hes, x)
nothing # hide
```

This linesearch problem only depends on the parameter ``\alpha``. We plot it:

```@setup ls_obj
alpha = 0.:.01:1.5

y = [problem.F(_alpha, params) for _alpha in alpha]
fig = Figure()
ax = Axis(fig[1, 1]; xlabel = L"\alpha", ylabel = L"f^\mathrm{ls}(\alpha)")
lines!(ax, alpha, y)

scatter!(ax, [α₁], [problem.F(α₁, params)]; color=mpurple, label=L"\alpha_1")
scatter!(ax, [α₂], [problem.F(α₂, params)]; color=morange, label=L"\alpha_2")
scatter!(ax, [α₃], [problem.F(α₃, params)]; color=mblue, label=L"\alpha_3")
scatter!(ax, [α₄], [problem.F(α₄, params)]; color=mgreen, label=L"\alpha_4")
scatter!(ax, [α₅], [problem.F(α₅, params)]; color=mred, label=L"\alpha_5")

axislegend(ax)

save("ls_backtracking_2d_plot_light.png", fig)
save("ls_backtracking_2d_plot_dark.png", fig)
nothing
```

![](ls_backtracking_2d_plot_light.png)
![](ls_backtracking_2d_plot_dark.png)

## [Example](@id sdc_example)

We show how to use line searches in `SimpleSolvers` to solve a simple toy problem[^3]:

[^3]: Also compare this to the case of the [static line search](@ref static_example).

```@example ls_obj
using SimpleSolvers # hide

ls_method = Backtracking()
nothing # hide
```

`SimpleSolvers` contains a function [`SimpleSolvers.linesearch_problem`](@ref) that allocates a [`LinesearchProblem`](@ref) that only depends on ``\alpha``:

We now use this to compute a *backtracking line search*:

```@example ls_obj
ls = Linesearch(problem, ls_method)
α = 50.
αₜ = solve(ls, α, params)
```

And we check whether the [`SufficientDecreaseCondition`](@ref) is satisfied:
```@example ls_obj
sdc = SufficientDecreaseCondition(c₁, problem.F(0., params), problem.D(0., params), alpha -> problem.F(alpha, params))
sdc(αₜ)
```

Similarly for the [`CurvatureCondition`](@ref):

```@example ls_obj
using SimpleSolvers: CurvatureCondition # hide
c₂ = .9
cc = CurvatureCondition(c₂, problem.D(0., params), alpha -> problem.D(alpha, params))
cc(αₜ)
```
