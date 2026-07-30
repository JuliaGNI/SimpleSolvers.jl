# Backtracking Line Search

A *backtracking line search method* determines the amount to move in a given search direction by iteratively decreasing a step size ``\alpha`` until an acceptable level is reached. In `SimpleSolvers` we use the [sufficient decrease condition](@ref "The Sufficient Decrease Condition") to quantify this *acceptable level*. The sufficient decrease condition is also referred to as the *Armijo condition* and together with the [curvature condition](@ref "The Curvature Condition") it forms the *Wolfe conditions*[^1] [nocedal2006numerical](@cite). 

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
problem = linesearch_problem(s)
params = (x = state.x, parameters = _params)
sdc = SufficientDecreaseCondition(c₁, problem.F(0., params), problem.D(0., params), alpha -> problem.F(alpha, params))

# check different values
α₁, α₂, α₃, α₄, α₅ = .09, .4, 0.7, 1., 1.3

using CairoMakie, LaTeXStrings
mred = RGBf(214 / 256, 39 / 256, 40 / 256)
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256)
mgreen = RGBf(44 / 256, 160 / 256, 44 / 256)
mblue = RGBf(31 / 256, 119 / 256, 180 / 256)
morange = RGBf(255 / 256, 127 / 256, 14 / 256)

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

## Stagnation at the round-off floor

The sufficient decrease condition demands a decrease *proportional to* ``\alpha``:

```math
\varphi(\alpha) \leq \varphi(0) + c_1\alpha\varphi'(0),
```

so once ``c_1\alpha|\varphi'(0)|`` drops below one unit in the last place of ``\varphi(0)``,
the right-hand side rounds back up to ``\varphi(0)`` exactly and the test degenerates to
``\varphi(\alpha) \leq \varphi(0)``. A merit that has reached its own round-off floor — think
of ``\|F\|^2`` for a residual that is already pure rounding noise, which is the normal state
of affairs at the end of a converged solve — then passes or fails that test essentially at
random. Shrinking ``\alpha`` cannot recover from this: below the round-off scale of ``x`` the
trial point stops differing from the base point altogether, and ``\varphi(\alpha)`` is
*bit-identical* to ``\varphi(0)``.

`Backtracking` therefore takes the round-off resolution of the merit,
``\tau = `` `τ_ulps` ``\cdot\,\mathrm{ulp}(\varphi(0))`` (see
[`SimpleSolvers.armijo_tolerance`](@ref)), slackens the condition to

```math
\varphi(\alpha) \leq \min\{\varphi(0),\ \varphi(0) + c_1\alpha\varphi'(0) + \tau\},
```

and stops at the smallest step that ``\tau`` can still resolve,
``\alpha_\mathrm{min} = \tau/(c_1|\varphi'(0)|)`` (see
[`SimpleSolvers.backtracking_αmin`](@ref)). Because ``\alpha_\mathrm{min}`` is a factor
``2\cdot`` `τ_ulps` above the step at which the rounding degeneracy sets in, the search stays
clear of that region — *unless* ``\alpha_\mathrm{min}``'s upper clamp at
``\sqrt{\mathrm{eps}(T)}`` binds, which it does for a very flat merit in double precision and for
essentially any merit in `Float16` (the clamp is there so that a nearly flat but genuine merit is
still searched at all).

The ``\min`` against ``\varphi(0)`` is what makes those trial steps harmless. Where the
right-hand side has degenerated, the condition reduces to plain monotonicity
``\varphi(\alpha) \leq \varphi(0)``: it can accept a non-increase but never an increase, and such
an accept is classified `LINESEARCH_FLOOR` rather than reported as a decrease. Without the
``\min`` the allowance would license a step whose merit is up to ``\tau`` *above* ``\varphi(0)``
— irrelevant at ``10^{-16}`` relative in double precision, but ``3.9\cdot10^{-3}`` in `Float16`,
twenty times the ``2c_1`` the condition demands at ``\alpha = 1``.

The situation is reported rather than hidden. [`solve_with_status`](@ref) returns a
[`LinesearchStatus`](@ref) whose [`LinesearchOutcome`](@ref) distinguishes a step that
genuinely decreased the merit (`LINESEARCH_DECREASED`) from one accepted only because nothing
*can* decrease it (`LINESEARCH_FLOOR`):

```@example floor
using SimpleSolvers
using SimpleSolvers: outcome, trials, steplength, issufficient, isfloor

# a merit that is pure round-off noise: every α > 0 lands one ulp above φ(0)
noise = LinesearchProblem{Float64}((α, _) -> α > 0 ? nextfloat(1.0) : 1.0, (α, _) -> -2.0)
ls = Linesearch(noise, Backtracking(); verbosity = 0)
st = solve_with_status(ls, 1.0)
(outcome(st), trials(st), steplength(st), st.αmin)
```

```@example floor
isfloor(st), issufficient(st)
```

A `LINESEARCH_FLOOR` outcome is *not* an error: it says that no line search can make progress
at this point, which is normally a statement about the problem rather than about the search.
A [`NonlinearSolver`](@ref) treats it as a stalled step (see
[`SimpleSolvers.stalled_step`](@ref)) and stops after `max_stalls` of them, reporting the
residual it did achieve against the tolerance that was requested — the usual cause being an
`f_abstol` below the residual's own round-off floor. See [`Options`](@ref).
