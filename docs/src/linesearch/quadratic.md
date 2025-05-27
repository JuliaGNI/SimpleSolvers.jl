# Quadratic Line Search

Quadratic [line search](@ref "Line Search") is based on making a quadratic approximation to an objective and then pick the minimum of this quadratic approximation as the next iteration of ``\alpha``.

The quadratic polynomial is built the following way[^1]:

[^1]: This is different from the [Bierlaire quadratic polynomial](@ref "Bierlaire Quadratic Line Search") described in [bierlaire2015optimization](@cite).

```math
p(\alpha) = f^\mathrm{ls}(0) + (f^\mathrm{ls})'(0)\alpha + p_2\alpha^2,
```

and we also call ``p_0:=f^\mathrm{ls}(0)`` and ``p_1:=(f^\mathrm{ls})'(0)``. The coefficient ``p_2`` is then determined the following way:
- take a value ``\alpha`` (typically initialized as [`SimpleSolvers.DEFAULT_ARMIJO_α₀`](@ref)) and compute ``y = f^\mathrm{ls}(\alpha)``,
- set ``p_2 \gets \frac{(y^2 - p_0 - p_1\alpha)}{\alpha^2}.``

After the polynomial is found we then take its minimum (analogously to the [Bierlaire quadratic line search](@ref "Bierlaire Quadratic Line Search")) and check if it satisfies the [sufficient decrease condition](@ref "The Sufficient Decrease Condition"). If it does not satisfy this condition we repeat the process, but with the current ``\alph`` as the starting point for the line search (instead of the initial [`SimpleSolvers.DEFAULT_ARMIJO_α₀`](@ref)).

## Example

Here we treat the following problem:

```@example quadratic
f(x::Union{T, Vector{T}}) where {T<:Number} = exp.(x) .* (x .^ 3 - 5x + 2x) .+ 2one(T)
f!(y::AbstractVector{T}, x::AbstractVector{T}) where {T} = y .= f.(x)
nothing # hide
```

```@setup quadratic
using CairoMakie
mred = RGBf(214 / 256, 39 / 256, 40 / 256)
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256)
mgreen = RGBf(44 / 256, 160 / 256, 44 / 256)
mblue = RGBf(31 / 256, 119 / 256, 180 / 256)
morange = RGBf(255 / 256, 127 / 256, 14 / 256)

fig_initial = Figure()
ax_initial = Axis(fig_initial[1, 1])
x = -1.:.01:2.
lines!(ax_initial, x, f.(x); label = L"f(x)")
axislegend(ax_initial)
save("f.png", fig_initial)
nothing # hide
```

![](f.png)

We now want to use quadratic line search to find the root of this function starting at ``x = 0``. We compute the Jacobian of ``f`` and initialize a [line search objective](@ref "Line Search Objective"):

```@example quadratic
using SimpleSolvers
using SimpleSolvers: update!, compute_jacobian!, factorize!, linearsolver, jacobian, cache, linesearch_objective, direction, determine_initial_α # hide
using LinearAlgebra: rmul!, ldiv! # hide
using Random # hide
Random.seed!(123) # hide

j!(j::AbstractMatrix{T}, x::AbstractVector{T}) where {T} = SimpleSolvers.ForwardDiff.jacobian!(j, f, x)
x = [0.]
# allocate solver
solver = NewtonSolver(x, f(x); F = f)
# initialize solver
update!(solver, x)
compute_jacobian!(solver, x, j!; mode = :function)

# compute rhs
f!(cache(solver).rhs, x)
rmul!(cache(solver).rhs, -1)

# multiply rhs with jacobian
factorize!(linearsolver(solver), jacobian(solver))
ldiv!(direction(cache(solver)), linearsolver(solver), cache(solver).rhs)
nls = NonlinearSystem(f, x)
ls_obj = linesearch_objective(nls, cache(solver))
fˡˢ = ls_obj.F
∂fˡˢ∂α = ls_obj.D
nothing # hide
```

```@setup quadratic
fig = Figure()
ax = Axis(fig[1, 1])
alpha = -2.:.01:2.
lines!(ax, alpha, fˡˢ.(alpha); label = L"f^\mathrm{ls}(\alpha)")
axislegend(ax)
save("f_ls.png", fig)
nothing # hide
```

![](f_ls.png)

The first two coefficient of the polynomial ``p`` (i.e. ``p_1`` and ``p_2``) are easy to compute:

```@example quadratic
p₀ = fˡˢ(0.)
p₁ = ∂fˡˢ∂α(0.)
nothing # hide
```

### Initializing ``\alpha``

In order to compute ``p_2`` we first have to initialize ``\alpha``. We start by *guessing* an initial ``\alpha`` as [`SimpleSolvers.DEFAULT_ARMIJO_α₀`](@ref). If this initial alpha does not satisfy the [`SimpleSolvers.BracketMinimumCriterion`](@ref), i.e. it holds that ``f^\mathrm{ls}(\alpha_0) > f^\mathrm{ls}(0)``, we call [`SimpleSolvers.bracket_minimum_with_fixed_point`](@ref) (similarly to calling [`SimpleSolvers.bracket_minimum`](@ref) for [standard bracketing](@ref "Bracketing")). 

Looking at [`SimpleSolvers.DEFAULT_ARMIJO_α₀`](@ref), we see that the [`SimpleSolvers.BracketMinimumCriterion`](@ref) is not satisfied:

```@setup quadratic
fig = Figure()
ax = Axis(fig[1, 1]; xlabel = L"\alpha")
alpha = -2.:.01:2.
lines!(ax, alpha, fˡˢ.(alpha); label = L"f^\mathrm{ls}(\alpha)")
scatter!(ax, SimpleSolvers.DEFAULT_ARMIJO_α₀, fˡˢ(SimpleSolvers.DEFAULT_ARMIJO_α₀); label = L"f^\mathrm{ls}(\alpha_{0, \mathrm{DEFAULT}})", color = mred)
axislegend(ax)
save("f_ls_daa.png", fig)
nothing # hide
```
![](f_ls_daa.png)

We therefore see that calling [`SimpleSolvers.determine_initial_α`](@ref) returns a different ``\alpha`` (the result of calling [`SimpleSolvers.bracket_minimum_with_fixed_point`](@ref)):

```@example quadratic
α₀ = determine_initial_α(ls_obj, SimpleSolvers.DEFAULT_ARMIJO_α₀)
```

```@setup quadratic
fig = Figure()
ax = Axis(fig[1, 1]; xlabel = L"\alpha")
alpha = -2.:.01:2.
lines!(ax, alpha, fˡˢ.(alpha); label = L"f^\mathrm{ls}(\alpha)")
scatter!(ax, α₀, fˡˢ(α₀); label = L"\alpha_0", color = mred)
axislegend(ax)
save("f_ls_a0.png", fig)
nothing # hide
```
![](f_ls_a0.png)

We can now finally compute ``p_2`` and determine the minimum of the polynomial:

```@example quadratic
y = fˡˢ(α₀)
p₂ = (y - p₀ - p₁*α₀) / α₀^2
p(α) = p₀ + p₁ * α + p₂ * α^2
αₜ = -p₁ / (2p₂)
```

When using [`SimpleSolvers.QuadraticState`](@ref) we in addition call [`SimpleSolvers.adjust_α`](@ref):

```@example quadratic
using SimpleSolvers: adjust_α # hide
α₁ = adjust_α(αₜ, α₀)
```

```@setup quadratic
using CairoMakie
mred = RGBf(214 / 256, 39 / 256, 40 / 256)
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256)
mgreen = RGBf(44 / 256, 160 / 256, 44 / 256)
mblue = RGBf(31 / 256, 119 / 256, 180 / 256)
morange = RGBf(255 / 256, 127 / 256, 14 / 256)

fig = Figure()
ax = Axis(fig[1, 1])
alpha = -2.:.01:2.
lines!(ax, alpha, fˡˢ.(alpha); label = L"f^\mathrm{ls}(\alpha)")
lines!(ax, alpha, p.(alpha); label = L"p^{(1)}(\alpha)")
scatter!(ax, α₁, p(α₁); color = mred, label = L"\alpha_1")
axislegend(ax)
save("f_ls1.png", fig)
nothing # hide
```

![](f_ls1.png)

We now check wether ``\alpha_1`` satisfies the [sufficient decrease condition](@ref "The Sufficient Decrease Condition"):

```@example quadratic
using SimpleSolvers: DEFAULT_WOLFE_c₁, SufficientDecreaseCondition # hide
sdc = SufficientDecreaseCondition(DEFAULT_WOLFE_c₁, 0., fˡˢ(0.), derivative(ls_obj, 0.), 1., ls_obj)
@assert sdc(α₁) # hide
sdc(α₁)
```

We now move the original ``x`` in the Newton direction with step length ``\alpha_1`` by using [`SimpleSolvers.compute_new_iterate`](@ref):

```@example quadratic
using SimpleSolvers: compute_new_iterate # hide
x .= compute_new_iterate(x, α₁, direction(cache(solver)))
```

```@setup quadratic
scatter!(ax_initial, x, f(x); color = mred, label = L"x^\mathrm{update}")
axislegend(ax_initial)
save("f_with_iterate.png", fig_initial)
nothing # hide
```
![](f_with_iterate.png)

And we see that we already very close to the root.

## Example for Optimization

We look again at the same example as before, but this time we want to find a minimum and not a root. We hence use [`SimpleSolvers.linesearch_objective`](@ref) not for a [`NewtonSolver`](@ref), but for an [`Optimizer`](@ref):

```@example quadratic
using SimpleSolvers: NewtonOptimizerCache, initialize!, gradient

x₀, x₁ = [0.], x
obj = MultivariateObjective(sum∘f, x₀)
gradient!(obj, x₀)
value!(obj, x₀)
_cache = NewtonOptimizerCache(x₀)
hess = Hessian(obj, x₀; mode = :autodiff)
update!(hess, x₀)
update!(_cache, x₀, gradient(obj), hess)
gradient!(obj, x₁)
value!(obj, x₁)
update!(hess, x₁)
update!(_cache, x₁, gradient(obj), hess)
ls_obj = linesearch_objective(obj, _cache)

fˡˢ = ls_obj.F
∂fˡˢ∂α = ls_obj.D
nothing # hide
```

```@setup quadratic
fig = Figure()
ax = Axis(fig[1, 1])
alpha = -2.:.01:2.
lines!(ax, alpha, fˡˢ.(alpha); label = L"f^\mathrm{ls}_\mathrm{opt}(\alpha)")
axislegend(ax)
save("f_ls_optimizer.png", fig)
nothing # hide
```

![](f_ls_optimizer.png)

!!! info
    Note the different shape of the line search objective in the case of the optimizer, especially that the line search objective can take negative values in this case!

We now again want to find the minimum with quadratic line search and repeat the procedure above:

```@example quadratic
p₀ = fˡˢ(0.)
```

```@example quadratic
p₁ = ∂fˡˢ∂α(0.)
```

```@example quadratic
α₀ = determine_initial_α(ls_obj, SimpleSolvers.DEFAULT_ARMIJO_α₀)
y = fˡˢ(α₀)
p₂ = (y - p₀ - p₁*α₀) / α₀^2
p(α) = p₀ + p₁ * α + p₂ * α^2
αₜ = -p₁ / (2p₂)
α₁ = adjust_α(αₜ, α₀)
```

```@setup quadratic
using CairoMakie
mred = RGBf(214 / 256, 39 / 256, 40 / 256)
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256)
mgreen = RGBf(44 / 256, 160 / 256, 44 / 256)
mblue = RGBf(31 / 256, 119 / 256, 180 / 256)
morange = RGBf(255 / 256, 127 / 256, 14 / 256)

fig = Figure()
ax = Axis(fig[1, 1])
alpha = -3.:.01:2.
lines!(ax, alpha, fˡˢ.(alpha); label = L"f^\mathrm{ls}_\mathrm{opt}(\alpha)")
lines!(ax, alpha, p.(alpha); label = L"p^{(1)}(\alpha)")
scatter!(ax, α₁, p(α₁); color = mred, label = L"\alpha_1")
axislegend(ax)
save("f_ls_opt1.png", fig)
nothing # hide
```

![](f_ls_opt1.png)

What we see here is that we do not use ``\alpha_t = -p_1 / (2p_2)`` as [`SimpleSolvers.adjust_α`](@ref) instead picks the left point in the interval ``[\sigma_0\alpha_0, \sigma_1\alpha_0]`` as the change computed with ``\alpha_t`` would be too small.

We now again move the original ``x`` in the Newton direction with step length ``\alpha_1``:

```@example quadratic
x .= compute_new_iterate(x, α₁, direction(_cache))
```

```@setup quadratic
fig = Figure()
ax = Axis(fig[1, 1])
x_array = -1.:.01:2.
lines!(ax, x_array, f.(x_array); label = L"f(x)")
scatter!(ax, x, f(x); color = mred, label = L"x^\mathrm{update}")
axislegend(ax)
save("f_with_iterate_opt.png", fig)
nothing # hide
```
![](f_with_iterate_opt.png)

We make another iteration:
```@example quadratic
gradient!(obj, x)
value!(obj, x)
update!(hess, x)
update!(_cache, x, gradient(obj), hess)
ls_obj = linesearch_objective(obj, _cache)

fˡˢ = ls_obj.F
∂fˡˢ∂α = ls_obj.D
p₀ = fˡˢ(0.)
p₁ = ∂fˡˢ∂α(0.)
α₀⁽²⁾ = determine_initial_α(ls_obj, SimpleSolvers.DEFAULT_ARMIJO_α₀)
y = fˡˢ(α₀)
p₂ = (y - p₀ - p₁*α₀⁽²⁾) / α₀⁽²⁾^2
p(α) = p₀ + p₁ * α + p₂ * α^2
αₜ = -p₁ / (2p₂)
```

```@example quadratic
α₂ = adjust_α(αₜ, α₀⁽²⁾)
```

We see that for ``\alpha_2`` (as opposed to ``\alpha_1``) we have ``\alpha_2 = \alpha_t`` as ``\alpha_t`` is in (this is what [`SimpleSolvers.adjust_α`](@ref) checks for):

```@example quadratic
using SimpleSolvers: DEFAULT_ARMIJO_σ₀, DEFAULT_ARMIJO_σ₁ # hide
(DEFAULT_ARMIJO_σ₀ * α₀⁽²⁾, DEFAULT_ARMIJO_σ₁ * α₀⁽²⁾)
```

```@example quadratic
x .= compute_new_iterate(x, α₂, direction(_cache))
```

```@setup quadratic
fig = Figure()
ax = Axis(fig[1, 1])
x_array = -1.:.01:2.
lines!(ax, x_array, f.(x_array); label = L"f(x)")
scatter!(ax, x, f(x); color = mred, label = L"x^\mathrm{update}")
axislegend(ax)
save("f_with_iterate_opt2.png", fig)
nothing # hide
```
![](f_with_iterate_opt2.png)

We finally compute a third iterate:
```@example quadratic
gradient!(obj, x)
value!(obj, x)
update!(hess, x)
update!(_cache, x, gradient(obj), hess)
ls_obj = linesearch_objective(obj, _cache)

fˡˢ = ls_obj.F
∂fˡˢ∂α = ls_obj.D
p₀ = fˡˢ(0.)
p₁ = ∂fˡˢ∂α(0.)
α₀⁽³⁾ = determine_initial_α(ls_obj, SimpleSolvers.DEFAULT_ARMIJO_α₀)
y = fˡˢ(α₀)
p₂ = (y - p₀ - p₁*α₀⁽³⁾) / α₀^2
p(α) = p₀ + p₁ * α + p₂ * α^2
αₜ = -p₁ / (2p₂)
α₃ = adjust_α(αₜ, α₀⁽³⁾)
```

```@example quadratic
x .= compute_new_iterate(x, α₃, direction(_cache))
```

```@setup quadratic
fig = Figure()
ax = Axis(fig[1, 1])
x_array = -1.:.01:2.
lines!(ax, x_array, f.(x_array); label = L"f(x)")
scatter!(ax, x, f(x); color = mred, label = L"x^\mathrm{update}")
axislegend(ax)
save("f_with_iterate_opt3.png", fig)
nothing # hide
```
![](f_with_iterate_opt3.png)

## Example II

Here we consider the same example as when discussing the [Bierlaire quadratic line search](@ref "Bierlaire Quadratic Line Search").

```@setup II
using SimpleSolvers
using SimpleSolvers: update!, compute_jacobian!, factorize!, linearsolver, jacobian, cache, linesearch_objective, direction
using LinearAlgebra: rmul!, ldiv!
using Random
Random.seed!(1234)

f(x::T) where {T<:Number} = exp(x) * (x ^ 3 - 5x ^ 2 + 2x) + 2one(T)
f(x::AbstractArray{T}) where {T<:Number} = exp.(x) .* (.5 * (x .^ 3) - 5 * (x .^ 2) + 2x) .+ 2one(T)
f!(y::AbstractVector{T}, x::AbstractVector{T}) where {T} = y .= f.(x)
j!(j::AbstractMatrix{T}, x::AbstractVector{T}) where {T} = SimpleSolvers.ForwardDiff.jacobian!(j, f!, similar(x), x)
x = -10 * rand(1)
solver = NewtonSolver(x, f.(x); F = f)
update!(solver, x)
compute_jacobian!(solver, x, j!; mode = :function)

# compute rhs
f!(cache(solver).rhs, x)
rmul!(cache(solver).rhs, -1)

# multiply rhs with jacobian
factorize!(linearsolver(solver), jacobian(solver))
ldiv!(direction(cache(solver)), linearsolver(solver), cache(solver).rhs)

nls = NonlinearSystem(f, x)
nothing # hide
```

```@example II
ls_obj = linesearch_objective(nls, cache(solver))
fˡˢ = ls_obj.F
∂fˡˢ∂α = ls_obj.D
nothing # hide
```

We now try to find a minimum of ``f^\mathrm{ls}`` with quadratic line search. For this we first need to find a bracket; we again do this with [`SimpleSolvers.bracket_minimum_with_fixed_point`](@ref)[^2]:

[^2]: Here we use [`SimpleSolvers.bracket_minimum_with_fixed_point`](@ref) directly instead of using [`SimpleSolvers.determine_initial_α`](@ref).

```@example II
(a, b) = SimpleSolvers.bracket_minimum_with_fixed_point(fˡˢ, 0.)
```

We plot the bracket:

```@example II
using CairoMakie
mred = RGBf(214 / 256, 39 / 256, 40 / 256)
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256)
mgreen = RGBf(44 / 256, 160 / 256, 44 / 256)
mblue = RGBf(31 / 256, 119 / 256, 180 / 256)
morange = RGBf(255 / 256, 127 / 256, 14 / 256)

fig = Figure()
ax = Axis(fig[1, 1])
alpha = -2.5:.01:3.
lines!(ax, alpha, fˡˢ.(alpha); label = L"f^\mathrm{ls}(\alpha)")
scatter!(ax, a, fˡˢ(a); color = mred, label = L"a")
scatter!(ax, b, fˡˢ(b); color = mpurple, label = L"b")
ylims!(ax, (-1., 6.))
axislegend(ax)
save("f_ls_1.png", fig)
nothing # hide
```

![](f_ls_1.png)

We now build the polynomial:

```@example II
p₀ = fˡˢ(a)
p₁ = ∂fˡˢ∂α(a)
y = fˡˢ(b)
p₂ = (y - p₀ - p₁*b) / b^2
p(α) = p₀ + p₁ * α + p₂ * α^2
nothing # hide
```

and compute its minimum:

```@example II
αₜ = -p₁ / (2p₂)
```

```@example II
lines!(ax, alpha, p.(alpha); label = L"p(\alpha)")
scatter!(ax, αₜ, p(αₜ); label = L"\alpha_t")
ylims!(ax, (-1., 6.))
axislegend(ax)
save("f_ls_2.png", fig)
nothing # hide
```

![](f_ls_2.png)

We now set ``a \gets \alpha_t`` and perform another iteration:

```@example II
(a, b) = SimpleSolvers.bracket_minimum_with_fixed_point(fˡˢ, αₜ)
```

We again build the polynomial:

```@example II
p₀ = fˡˢ(a)
p₁ = ∂fˡˢ∂α(a)
y = fˡˢ(b)
p₂ = (y - p₀ - p₁*(b-a)) / (b-a)^2
p(α) = p₀ + p₁ * (α-a) + p₂ * (α-a)^2
nothing # hide
```

and compute its minimum:

```@example II
αₜ = -p₁ / (2p₂) + a
```

```@setup II
fig = Figure()
ax = Axis(fig[1, 1])
alpha = -2.5:.01:3.
lines!(ax, alpha, fˡˢ.(alpha); label = L"f^\mathrm{ls}(\alpha)")
scatter!(ax, a, fˡˢ(a); color = mred, label = L"a")
scatter!(ax, b, fˡˢ(b); color = mpurple, label = L"b")
axislegend(ax)
lines!(ax, alpha, p.(alpha); label = L"p(\alpha)")
scatter!(ax, αₜ, p(αₜ); label = L"\alpha_t")
# ylims!(ax, (-1., 6.))
axislegend(ax)
save("f_ls_3.png", fig)
nothing # hide
```

![](f_ls_3.png)