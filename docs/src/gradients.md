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

Instead of calling `GradientAutodiff(f, x)` we can equivalently do:

```@example gradient
grad = Gradient(f, x; mode = :autodiff)
```

When calling an instance of [`Gradient`](@ref) we can use the functions [`gradient`](@ref) and [`gradient!`](@ref)[^1]:

[^1]: Internally these functions call functors that are implemented for the individual `struct`s derived from [`Gradient`](@ref), but for consistency (especially with regards to [`OptimizerProblem`](@ref)s) we recommend using the functions [`gradient`](@ref) and [`gradient!`](@ref).

```@example gradient
gradient(x, grad)
```