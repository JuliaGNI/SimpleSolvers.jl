# Quadratic Line Search

Quadratic [line search](@ref "Line Search") is based on making a quadratic approximation to an optimizer problem and then pick the minimum of this quadratic approximation as the next iteration of ``\alpha``.

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
F!(y::AbstractVector{T}, x::AbstractVector{T}, params) where {T} = f!(y, x)
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
x = [0.]
scatter!(ax_initial, x, f.(x); label = L"x_0", color = :red)
axislegend(ax_initial)
save("f.png", fig_initial)
nothing # hide
```

![](f.png)

We now want to use quadratic line search to find the root of this function starting at ``x = 0``. We compute the Jacobian of ``f`` and initialize a [line search problem](@ref "Line Search Problem"):

```@example quadratic
using SimpleSolvers
using SimpleSolvers: factorize!, update!, linearsolver, jacobian, jacobian!, cache, linesearch_problem, direction, determine_initial_α # hide
using LinearAlgebra: rmul!, ldiv! # hide
using Random # hide
Random.seed!(123) # hide

function J!(j::AbstractMatrix{T}, x::AbstractVector{T}, params) where {T}
    SimpleSolvers.ForwardDiff.jacobian!(j, f, x)
end

# allocate solver
solver = NewtonSolver(x, f(x); F = F!, DF! = J!)
# initialize solver
params = nothing
update!(solver, x, params)
jacobian!(solver, x, params)

# compute rhs
F!(cache(solver).rhs, x, params)
rmul!(cache(solver).rhs, -1)

# multiply rhs with jacobian
factorize!(linearsolver(solver), jacobian(solver))
ldiv!(direction(cache(solver)), linearsolver(solver), cache(solver).rhs)
nlp = NonlinearProblem(F!, J!, x, f(x))
ls_obj = linesearch_problem(nlp, Jacobian(solver), cache(solver), params)
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

!!! info
    The second plot shows the optimization problem for the ideal step length, where we start from ``x_0`` and proceed in the Newton direction. In the following we want to determine its minimum by fitting a quadratic polynomial, i.e. fitting ``p``.

The first two coefficient of the polynomial ``p`` (i.e. ``p_1`` and ``p_2``) are easy to compute:

```@example quadratic
p₀ = fˡˢ(0.)
```

```@example quadratic
p₁ = ∂fˡˢ∂α(0.)
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

When using `QuadraticState` we in addition call [`SimpleSolvers.adjust_α`](@ref):

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

We now move the original ``x`` in the Newton direction with step length ``\alpha_1`` by using [`SimpleSolvers.compute_new_iterate!`](@ref):

```@example quadratic
using SimpleSolvers: compute_new_iterate! # hide
compute_new_iterate!(x, α₁, direction(cache(solver)))
```

```@setup quadratic
scatter!(ax_initial, x, f(x); color = mpurple, label = L"x^\mathrm{update}")
axislegend(ax_initial; merge = true, unique = true)
save("f_with_iterate.png", fig_initial)
nothing # hide
```
![](f_with_iterate.png)

And we see that we already very close to the root.

## Example for Optimization

We look again at the same example as before, but this time we want to find a minimum and not a root. We hence use [`SimpleSolvers.linesearch_problem`](@ref) not for a [`NewtonSolver`](@ref), but for an [`Optimizer`](@ref):

```@example quadratic
using SimpleSolvers: NewtonOptimizerCache, initialize!, gradient

x₀, x₁ = [0.], x
obj = OptimizerProblem(sum∘f, x₀)
grad = GradientAutodiff{Float64}(obj.F, length(x))
_cache = NewtonOptimizerCache(x₀)
state = NewtonOptimizerState(x₀)
hess = HessianAutodiff(obj, x₀)
H = SimpleSolvers.alloc_h(x)
hess(H, x₀)
update!(_cache, state, grad, hess, x₀)
hess(H, x₁)
update!(_cache, state, grad, hess, x₁)
ls_obj = linesearch_problem(obj, grad, _cache, state)

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
    Note the different shape of the line search problem in the case of the optimizer, especially that the line search problem can take negative values in this case!

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
```

```@example quadratic
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
scatter!(ax, αₜ, p(αₜ); color = mpurple, label = L"\alpha_t")
scatter!(ax, α₁, p(α₁); color = mred, label = L"\alpha_1")
axislegend(ax)
save("f_ls_opt1.png", fig)
nothing # hide
```

![](f_ls_opt1.png)

What we see here is that we do not use ``\alpha_t = -p_1 / (2p_2)`` as [`SimpleSolvers.adjust_α`](@ref) instead picks the left point in the interval ``[\sigma_0\alpha_0, \sigma_1\alpha_0]`` as the change computed with ``\alpha_t`` would be too small.

We now again move the original ``x`` in the Newton direction with step length ``\alpha_1``:

```@example quadratic
compute_new_iterate!(x, α₁, direction(_cache))
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
hess(H, x)
update!(_cache, state, grad, hess, x)
ls_obj = linesearch_problem(obj, grad, _cache, state)

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

```@setup quadratic
fig = Figure()
ax = Axis(fig[1, 1])
alpha = -15.:.01:2.
lines!(ax, alpha, fˡˢ.(alpha); label = L"f^\mathrm{ls}_\mathrm{opt}(\alpha)")
lines!(ax, alpha, p.(alpha); label = L"p^{(2)}(\alpha)")
scatter!(ax, αₜ, p(αₜ); color = mpurple, label = L"\alpha_t")
scatter!(ax, α₂, p(α₂); color = mred, label = L"\alpha_2")
axislegend(ax)
save("f_ls_opt2.png", fig)
nothing # hide
```

![](f_ls_opt2.png)

We see that for ``\alpha_2`` (as opposed to ``\alpha_1``) we have ``\alpha_2 = \alpha_t`` as ``\alpha_t`` is in (this is what [`SimpleSolvers.adjust_α`](@ref) checks for):

```@example quadratic
using SimpleSolvers: DEFAULT_ARMIJO_σ₀, DEFAULT_ARMIJO_σ₁ # hide
(DEFAULT_ARMIJO_σ₀ * α₀⁽²⁾, DEFAULT_ARMIJO_σ₁ * α₀⁽²⁾)
```

```@example quadratic
using SimpleSolvers: compute_new_iterate
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
hess(H, x)
update!(_cache, state, grad, hess, x)
ls_obj = linesearch_problem(obj, grad, _cache, state)

fˡˢ = ls_obj.F
∂fˡˢ∂α = ls_obj.D
p₀ = fˡˢ(0.)
p₁ = ∂fˡˢ∂α(0.)
α₀⁽³⁾ = determine_initial_α(ls_obj, SimpleSolvers.DEFAULT_ARMIJO_α₀)
y = fˡˢ(α₀)
p₂ = (y - p₀ - p₁*α₀⁽³⁾) / α₀^2
p(α) = p₀ + p₁ * α + p₂ * α^2
αₜ = -p₁ / (2p₂)
```

```@example quadratic
α₃ = adjust_α(αₜ, α₀⁽³⁾)
```

```@example quadratic
x .= compute_new_iterate(x, α₃, direction(_cache))
```

```@setup quadratic
fig = Figure()
ax = Axis(fig[1, 1])
x_array = -1.:.01:4.
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
using SimpleSolvers: jacobian!, factorize!, linearsolver, jacobian, cache, linesearch_problem, direction
using LinearAlgebra: rmul!, ldiv!
using Random
Random.seed!(1234)

f(x::T) where {T<:Number} = exp(x) * (x ^ 3 - 5x ^ 2 + 2x) + 2one(T)
f(x::AbstractArray{T}) where {T<:Number} = exp.(x) .* (.5 * (x .^ 3) - 5 * (x .^ 2) + 2x) .+ 2one(T)
f!(y::AbstractVector{T}, x::AbstractVector{T}) where {T} = y .= f.(x)
j!(j::AbstractMatrix{T}, x::AbstractVector{T}) where {T} = SimpleSolvers.ForwardDiff.jacobian!(j, f!, similar(x), x)
F!(y, x, params) = f!(y, x)
J!(j, x, params) = j!(j, x)

x = -10 * rand(1)
solver = NewtonSolver(x, f.(x); F = F!, DF! = J!)
params = nothing
update!(solver, x, params)
jacobian!(solver, x, params)

# compute rhs
f!(cache(solver).rhs, x)
rmul!(cache(solver).rhs, -1)

# multiply rhs with jacobian
factorize!(linearsolver(solver), jacobian(solver))
ldiv!(direction(cache(solver)), linearsolver(solver), cache(solver).rhs)

nlp = NonlinearProblem(F!, J!, x, f(x))
nothing # hide
```

```@example II
ls_obj = linesearch_problem(nlp, JacobianFunction{Float64}(F!, J!), cache(solver), params)
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
alpha = -2:.01:6.
lines!(ax, alpha, fˡˢ.(alpha); label = L"f^\mathrm{ls}(\alpha)")
scatter!(ax, a, fˡˢ(a); color = mred, label = L"a")
scatter!(ax, b, fˡˢ(b); color = mpurple, label = L"b")
# ylims!(ax, (-1., 6.)) # hide
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
# ylims!(ax, (-1., 6.)) # hide
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