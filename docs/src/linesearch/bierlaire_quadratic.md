# Bierlaire Quadratic Line Search

In [bierlaire2015optimization](@cite) quadratic line search is defined as an interpolation between three points. For this consider

```@example bierlaire
using SimpleSolvers
using SimpleSolvers: factorize!, linearsolver, jacobian, jacobian!, cache, linesearch_problem, direction, NullParameters, NonlinearSolverState # hide
using LinearAlgebra: rmul!, ldiv! # hide
using Random # hide
Random.seed!(1234) # hide

f(x::T) where {T<:Number} = exp(x) * (T(.5) * x ^ 3 - 5x ^ 2 + 2x) + 2one(T)
f(x::AbstractArray{T}) where {T<:Number} = exp.(x) .* (T(.5) * (x .^ 3) - 5 * (x .^ 2) + 2x) .+ 2one(T)
f!(y::AbstractVector{T}, x::AbstractVector{T}) where {T} = y .= f.(x)
j!(j::AbstractMatrix{T}, x::AbstractVector{T}) where {T} = SimpleSolvers.ForwardDiff.jacobian!(j, f!, similar(x), x)
F!(y, x, params) = f!(y, x)
J!(j, x, params) = j!(j, x)
x = -10 * rand(1)
solver = NewtonSolver(x, f.(x); F = F!, DF! = J!)
params = NullParameters()
state = NonlinearSolverState(x)
update!(state, x, f(x), 0)
jacobian!(solver, x, params)

# compute rhs
f!(cache(solver).rhs, x)
rmul!(cache(solver).rhs, -1)

# multiply rhs with jacobian
factorize!(linearsolver(solver), jacobian(solver))
ldiv!(direction(cache(solver)), linearsolver(solver), cache(solver).rhs)

nlp = NonlinearProblem(F!, x, f(x))
ls_obj = linesearch_problem(nlp, Jacobian(solver), cache(solver), state, params)
fˡˢ = ls_obj.F
∂fˡˢ∂α = ls_obj.D
nothing # hide
```

For the Bierlaire quadratic line search we need three points: ``a``, ``b`` and ``c``:

```@example bierlaire
a, b, c = -2., 0.5, 2.5
nothing # hide
```

```@setup bierlaire
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
scatter!(ax, c, fˡˢ(c); color = morange, label = L"c")
axislegend(ax)
save("f_ls_bierlaire1.png", fig)
nothing # hide
```

![](f_ls_bierlaire1.png)

In the figure above we already plotted three points ``a``, ``b`` and ``c`` on whose basis a second-order polynomial will be built that should approximate ``f^\mathrm{ls}``.[^1] The polynomial is built with the ansatz:

[^1]: These points further need to satisfy ``f^\mathrm{ls}(a) > f^\mathrm{ls}(b) < f^\mathrm{ls}(c)``.

```math
p(\alpha) = \beta_1(\alpha - a)(x - b) + \beta_2(\alpha - a) + \beta_3(\alpha - b),
```

and by identifying 

```math
\begin{aligned}
p(a) & = f^\mathrm{ls}(a), \\
p(b) & = f^\mathrm{ls}(b), \\
p(c) & = f^\mathrm{ls}(c), \\
\end{aligned}
```

we get

```math
\begin{aligned}
\beta_1 & = \frac{(b - c)f^\mathrm{ls}(a) + (c - a)f^\mathrm{ls}(b) + (a - b)f^\mathrm{ls}(c)}{(a - b)(c - a)(c - b)}, \\ 
\beta_2 & = \frac{f^\mathrm{ls}(b)}{b - a}, \\
\beta_3 & = \frac{f^\mathrm{ls}(a)}{a - b}.
\end{aligned}
```

We can plot this polynomial:

```@setup bierlaire
β₁ = ((b - c) * fˡˢ(a) + (c - a) * fˡˢ(b) + (a - b) * fˡˢ(c)) / ((a - b) * (c - a) * (c - b))
β₂ = fˡˢ(b) / (b - a)
β₃ = fˡˢ(a) / (a - b)
p(α) = β₁ * (α - a) * (α - b) + β₂ * (α - a) + β₃ * (α - b)
lines!(ax, alpha, p.(alpha); color = mgreen, label = L"p(\alpha)")
axislegend(ax; position = :rt)
save("f_ls_bierlaire2.png", fig)
nothing
```

![](f_ls_bierlaire2.png)

We can now easily determine the minimum of the polynomial ``p``. It is:

```math
\chi = \frac{1}{2} \frac{ f^\mathrm{ls}(a) (b^2 - c^2) + f^\mathrm{ls}(b) (c^2 - a^2) + f^\mathrm{ls}(c) (a^2 - b^2) }{f^\mathrm{ls}(a) (b - c) + f^\mathrm{ls}(b) (c - a) + f^\mathrm{ls}(c) (a - b)}.
```

```@setup bierlaire
χ = .5 * ( fˡˢ(a) * (b^2 - c^2) + fˡˢ(b) * (c^2 - a^2) + fˡˢ(c) * (a^2 - b^2) ) / (fˡˢ(a) * (b - c) + fˡˢ(b) * (c - a) + fˡˢ(c) * (a - b))
scatter!(ax, χ, p(χ); color = mblue, label=L"\chi")
axislegend(ax; position = :rt)
save("f_ls_bierlaire3.png", fig)
```

![](f_ls_bierlaire3.png)

We now use this ``\chi`` to either replace ``a``, ``b`` or ``c`` and distinguish between the following four scenarios:
1. ``\chi > b`` and ``f^\mathrm{ls}(\chi) > f^\mathrm{ls}(b)`` ``\implies`` we replace ``c \gets \chi``,
2. ``\chi > b`` and ``f^\mathrm{ls}(\chi) \leq f^\mathrm{ls}(b)`` ``\implies`` we replace ``a, b \gets b, \chi``,
3. ``\chi \leq b`` and ``f^\mathrm{ls}(\chi) > f^\mathrm{ls}(b)`` ``\implies`` we replace ``a \gets \chi``,
4. ``\chi \leq b`` and ``f^\mathrm{ls}(\chi) \leq f^\mathrm{ls}(b)`` ``\implies`` we replace ``b, c \gets \chi, b``.

In our example we have the second case: ``\chi`` is to the right of ``b`` and ``f^\mathrm{ls}(\chi)`` is smaller than ``f(b)``. We therefore replace ``a`` with ``b`` and ``\b`` with ``\chi``. The new approximation is the following one:

```@setup bierlaire
fig = Figure()
ax = Axis(fig[1, 1])
alpha = -0.:.01:2.5
@assert b < χ
@assert f(b) > f(χ)
a = b
b = χ
lines!(ax, alpha, fˡˢ.(alpha); label = L"f^\mathrm{ls}(\alpha)")
scatter!(ax, a, fˡˢ(a); color = mred, label = L"a")
scatter!(ax, b, fˡˢ(b); color = mpurple, label = L"b")
scatter!(ax, c, fˡˢ(c); color = morange, label = L"c")
β₁ = ((b - c) * fˡˢ(a) + (c - a) * fˡˢ(b) + (a - b) * fˡˢ(c)) / ((a - b) * (c - a) * (c - b))
β₂ = fˡˢ(b) / (b - a)
β₃ = fˡˢ(a) / (a - b)
p(α) = β₁ * (α - a) * (α - b) + β₂ * (α - a) + β₃ * (α - b)
lines!(ax, alpha, p.(alpha); color = mgreen, label = L"p(\alpha)")
χ = .5 * ( fˡˢ(a) * (b^2 - c^2) + fˡˢ(b) * (c^2 - a^2) + fˡˢ(c) * (a^2 - b^2) ) / (fˡˢ(a) * (b - c) + fˡˢ(b) * (c - a) + fˡˢ(c) * (a - b))
scatter!(ax, χ, p(χ); color = mblue, label=L"\chi")
axislegend(ax; position = :rb)
save("f_ls_bierlaire4.png", fig)
```

![](f_ls_bierlaire4.png)

We again observe the second case. By replacing ``a, b \gets b, \chi`` we get:

```@setup bierlaire
fig = Figure()
ax = Axis(fig[1, 1])
alpha = .4:.01:2.5
@assert b < χ
@assert f(b) > f(χ)
a = b
b = χ
lines!(ax, alpha, fˡˢ.(alpha); label = L"f^\mathrm{ls}(\alpha)")
scatter!(ax, a, fˡˢ(a); color = mred, label = L"a")
scatter!(ax, b, fˡˢ(b); color = mpurple, label = L"b")
scatter!(ax, c, fˡˢ(c); color = morange, label = L"c")
β₁ = ((b - c) * fˡˢ(a) + (c - a) * fˡˢ(b) + (a - b) * fˡˢ(c)) / ((a - b) * (c - a) * (c - b))
β₂ = fˡˢ(b) / (b - a)
β₃ = fˡˢ(a) / (a - b)
p(α) = β₁ * (α - a) * (α - b) + β₂ * (α - a) + β₃ * (α - b)
lines!(ax, alpha, p.(alpha); color = mgreen, label = L"p(\alpha)")
χ = .5 * ( fˡˢ(a) * (b^2 - c^2) + fˡˢ(b) * (c^2 - a^2) + fˡˢ(c) * (a^2 - b^2) ) / (fˡˢ(a) * (b - c) + fˡˢ(b) * (c - a) + fˡˢ(c) * (a - b))
scatter!(ax, χ, p(χ); color = mblue, label=L"\chi")
axislegend(ax; position = :rb)
save("f_ls_bierlaire5.png", fig)
```

![](f_ls_bierlaire5.png)

We now observe the first case: ``\chi`` is to the left of ``b`` and ``f^\mathrm{ls}(\chi)`` is above ``f(b)``. Hence we replace ``b, c \gets \chi, b.`` A successive iteration yields:

```@setup bierlaire
@assert fˡˢ(χ) ≤ fˡˢ(b)
fig = Figure()
ax = Axis(fig[1, 1])
alpha = .45:.01:1.3
@assert b > χ
@assert f(b) < f(χ)
c = b
b = χ
lines!(ax, alpha, fˡˢ.(alpha); label = L"f^\mathrm{ls}(\alpha)")
scatter!(ax, a, fˡˢ(a); color = mred, label = L"a")
scatter!(ax, b, fˡˢ(b); color = mpurple, label = L"b")
scatter!(ax, c, fˡˢ(c); color = morange, label = L"c")
β₁ = ((b - c) * fˡˢ(a) + (c - a) * fˡˢ(b) + (a - b) * fˡˢ(c)) / ((a - b) * (c - a) * (c - b))
β₂ = fˡˢ(b) / (b - a)
β₃ = fˡˢ(a) / (a - b)
p(α) = β₁ * (α - a) * (α - b) + β₂ * (α - a) + β₃ * (α - b)
lines!(ax, alpha, p.(alpha); color = mgreen, label = L"p(\alpha)")
χ = .5 * ( fˡˢ(a) * (b^2 - c^2) + fˡˢ(b) * (c^2 - a^2) + fˡˢ(c) * (a^2 - b^2) ) / (fˡˢ(a) * (b - c) + fˡˢ(b) * (c - a) + fˡˢ(c) * (a - b))
# scatter!(ax, χ, p(χ); color = mblue, label=L"\chi")
axislegend(ax; position = :rt)
save("f_ls_bierlaire6.png", fig)
```

![](f_ls_bierlaire6.png)

!!! info
    After having computed ``\chi`` we further either shift it to the left or right depending on whether ``(c - b)`` or ``(b - a)`` is bigger respectively. The shift is made by either adding or subtracting the constant ``\varepsilon``.
Also see [`SimpleSolvers.DEFAULT_BIERLAIRE_ε`](@ref).