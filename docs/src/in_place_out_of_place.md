# What is In-Place and what is Out-Of-Place

In `SimpleSolvers` we almost always use in-place functions internally for performance, but let the user deal with out-of-place functions for ease of use.

# Example

```@example in_place
using SimpleSolvers

f(x) = sum(x.^2 .* exp.(-abs.(x)) + 2 * cos.(x) .* exp.(-x.^2))
nothing # hide
```

```@setup in_place
using CairoMakie
mred = RGBf(214 / 256, 39 / 256, 40 / 256)
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256)
mgreen = RGBf(44 / 256, 160 / 256, 44 / 256)
mblue = RGBf(31 / 256, 119 / 256, 180 / 256)
morange = RGBf(255 / 256, 127 / 256, 14 / 256)

fig = Figure()
ax = Axis(fig[1, 1]; xlabel = L"x", ylabel=L"f(x)")
x = -7.:.1:7.
lines!(ax, x, f.(x))
save("f.png", fig)
nothing # hide
```

![](f.png)

If we now allocate a [`MultivariateObjective`](@ref) based on this, we get a series of in-place functions based on this. For example [`value!`](@ref)[^1]:

[^1]: See the [section on objectives](@ref "Objectives") for an explanation of how to use [`value!`](@ref) and [`value`](@ref).

```@example in_place
x = [0.]
obj = MultivariateObjective(f, x)
y = [0.]
value!(obj, x)
@assert value(obj) == f(x) # hide
value(obj) == f(x)
```

To compute the derivative we can use [`gradient!`](@ref)[^2]:

[^2]: Note that we are using a [`MultivariateObjective`](@ref) and therefore [`gradient!`](@ref). A [`UnivariateObjective`](@ref) has to be used together with [`derivative`](@ref).

```@example in_place
x = [[x] for x in -7.:.1:7.]
y = Vector{Float64}[]
for x_sing in x
    gradient!(obj, x_sing)
    push!(y, copy(gradient(obj)))
end
nothing # hide
```

```@setup in_place
fig = Figure()
ax = Axis(fig[1, 1]; xlabel = L"x", ylabel=L"\frac{\partial{}f(x)}{\partial{}x}")

lines!(ax, reduce(vcat, x), reduce(vcat, y))
save("f_prime.png", fig)
nothing # hide
```

![](f_prime.png)

The idea is however that the user almost never used the in-place versions of these routines directly, but instead functions like [`solve!`](@ref) and [`value`](@ref), [`gradient`](@ref) etc. as a possible diagnostic.