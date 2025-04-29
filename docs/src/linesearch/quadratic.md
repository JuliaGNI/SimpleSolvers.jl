# Quadratic Line Search

Quadratic [line search](@ref "Line Search") is based on making a quadratic approximation to an objective and then pick the minimum of this quadratic approximation as the next iteration of ``\alpha``.

The quadratic polynomial is built the following way:

```math
p(\alpha) = f^\mathrm{ls}(0) + (f^\mathrm{ls})'(0)\alpha + p_2\alpha^2,
```

and we also call ``p_0:=f^\mathrm{ls}(0)`` and ``p_1:=(f^\mathrm{ls})'(0)``. The coefficient ``p_2`` is then determined the following way:
- take a value ``\alpha`` (typically initialized as [`SimpleSolvers.DEFAULT_ARMIJO_α₀`](@ref)) and compute ``y = f^\mathrm{ls}(\alpha)``,
- set ``p_2 \gets \frac{(y^2 - p_0 - p_1\alpha)}{\alpha^2}.``

## Example

```@example quadratic
using SimpleSolvers
using SimpleSolvers: update!, compute_jacobian!, factorize!, linearsolver, jacobian, cache, linesearch_objective, direction # hide
using LinearAlgebra: rmul!, ldiv! # hide
using Random # hide
Random.seed!(123) # hide

f(x::T) where {T<:Number} = exp(x) * (x ^ 3 - 5x + 2x) + 2one(T)
f!(y::AbstractVector{T}, x::AbstractVector{T}) where {T} = y .= f.(x)
j!(j::AbstractMatrix{T}, x::AbstractVector{T}) where {T} = SimpleSolvers.ForwardDiff.jacobian!(j, f!, similar(x), x)
x = rand(1)
solver = NewtonSolver(x, f.(x); F = f)
update!(solver, x)
compute_jacobian!(solver, x, j!; mode = :function)

factorize!(linearsolver(solver), jacobian(cache(solver)))

f!(cache(solver).rhs, x)
rmul!(cache(solver).rhs, -1)

# solve J δx = -f(x)
ldiv!(direction(cache(solver)), linearsolver(solver), cache(solver).rhs)
ls_obj = linesearch_objective(f!, JacobianFunction(j!, x), cache(solver))
nothing # hide
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
alpha = -1.:.01:2.
p₀ = ls_obj.F(0.)
p₁ = ls_obj.D(0.)
α₀ = SimpleSolvers.DEFAULT_ARMIJO_α₀
y = ls_obj.F(α₀)
p₂ = (y^2 - p₀ - p₁*α₀) / α₀^2
p(α) = p₀ + p₁ * α + p₂ * α^2
αₜ = -p₁ / (2p₂)
lines!(ax, alpha, ls_obj.F.(alpha); label = L"f^\mathrm{ls}(\alpha)")
scatter!(ax, αₜ, p(αₜ); color = mred)
axislegend(ax)
save("f_ls.png", fig)
nothing # hide
```

![](f_ls.png)

## Example in the Non-Convex Case

We give an example of a polynomial approximation for a function in a non-convex part of a function:

```@example quadratic
d(x) = SimpleSolvers.ForwardDiff.derivative(f, x)
p₀ = f(0.)
p₁ = d(0.)
α₀ = SimpleSolvers.DEFAULT_ARMIJO_α₀
y = f(α₀)
p₂ = (y^2 - p₀ - p₁*α₀) / α₀^2
p(α) = p₀ + p₁ * α + p₂ * α^2
nothing # hide
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
alpha = -1.:.01:2.
lines!(ax, alpha, f.(alpha); label = L"f(\alpha)")
lines!(ax, alpha, p.(alpha); label = L"p(\alpha)")
αₜ = -p₁ / (2p₂)
scatter!(ax, αₜ, p(αₜ); color = mred)
axislegend(ax)
save("f_v_p.png", fig)
nothing # hide
```

![](f_v_p.png)

red dot shown in the graph is ``(\alpha_t, p(\alpha_t))``. When calling the functor of [`SimpleSolvers.QuadraticState`](@ref) this ``\alpha_t`` is adjusted with the function [`SimpleSolvers.adjust_alpha`](@ref)[^1].

```@example quadratic
using SimpleSolvers: adjust_alpha
ls = SimpleSolvers.QuadraticState()
α₁ = adjust_alpha(ls, αₜ, α₀)
```

[^1]: The functor of [`SimpleSolvers.QuadraticState`](@ref) first determines ``p_0``, ``p_1``, ``p_2`` and successively ``\alpha_t``, and then uses [`SimpleSolvers.adjust_alpha`](@ref) to adjust this ``\alpha_t`` by relying on ``\sigma_0`` and ``\sigma_1`` (by default [`SimpleSolvers.DEFAULT_ARMIJO_σ₀`](@ref) and [`SimpleSolvers.DEFAULT_ARMIJO_σ₁`](@ref)).

This process of updating ``\alpha`` is now repeated iteratively:

```@example quadratic
y = f(α₁)
p₂ = (y^2 - p₀ - p₁*α₁) / α₁^2
p(α) = p₀ + p₁ * α + p₂ * α^2
nothing # hide
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
alpha = -1.:.01:2.
lines!(ax, alpha, f.(alpha); label = L"f(\alpha)")
lines!(ax, alpha, p.(alpha); label = L"p(\alpha)")
αₜ = -p₁ / (2p₂)
scatter!(ax, αₜ, p(αₜ); color = mred)
axislegend(ax)
save("f_v_p2.png", fig)
nothing # hide
```

![](f_v_p2.png)