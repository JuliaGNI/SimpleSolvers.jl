# Gradients

The supertype [`Gradient`](@ref) comprises different ways of taking gradients:
- [`GradientFunction`](@ref),
- [`GradientAutodiff`](@ref),
- [`GradientFiniteDifferences`](@ref).

We first start by showing [`GradientAutodiff`](@ref):

```@example gradient
using SimpleSolvers, Random; using SimpleSolvers: GradientAutodiff, Gradient, GradientFunction, GradientFiniteDifferences; Random.seed!(123) # hide
f(x::AbstractArray) = sum(x .^ 2)
x = rand(3)
grad = GradientAutodiff(f, x)
```

Every `struct` derived from [`Gradient`](@ref) (including [`GradientAutodiff`](@ref)) has an associated functor:

```@example gradient
grad(x)
```