# Objectives

A central object in `SimpleSolvers` are *objectives* (see [`SimpleSolvers.AbstractObjective`](@ref)). They are either [`SimpleSolvers.AbstractUnivariateObjective`](@ref)s or [`MultivariateObjective`](@ref)s. The goal of a *solver* (both [`LinearSolver`](@ref)s and [`NonlinearSolver`](@ref)s) is to make the objective have value zero. The goal of an [`Optimizer`](@ref) is to minimize a [`MultivariateObjective`](@ref).

## Examples

### Univariate Objectives

We can allocate a univariate objective with:

```@example objective; setup = (:using Random; Random.seed!(123))
using SimpleSolvers

f(x::Number) = tanh(x)
x = rand()
obj = UnivariateObjective(f, x)
```

Associated to [`UnivariateObjective`](@ref) are the following functions (amongst others):
- [`value`](@ref)
- [`value!`](@ref)
- [`value!!`](@ref)
- [`derivative`](@ref)
- [`derivative!`](@ref)
- [`derivative!!`](@ref)

The function [`value`](@ref) evaluates the objective at the provided input and increases the counter by 1:

```@example objective
y = value(obj, x)
```

We can check how the function call changed `obj`:

```@example objective
obj
```

The stored value for `f` has not been updated. In order to so we can call the in-place function [`value!`](@ref):

```@example objective
y = value!(obj, x)
obj
```

We further note that `SimpleSolvers` contains another function [`value!!`](@ref) that *forces evaluation*. This is opposed to [`value!`](@ref) which does not always force evaluation:

```@example objective
y = value!(obj, x)
obj
```

So [`value!`](@ref) first checks if the objective `obj` has already been called on `x`. In order to force another evaluation we can write:

```@example objective
y = value!!(obj, x)
obj
```

The function [`value`](@ref) can also be called without additional input arguments:

```@example objective
value(obj)
```

But then the associated function is not called again (calling [`value`](@ref) this way does not increase the counter):

```@example objective
obj
```

An equivalent relationship exists between the functions [`derivative`](@ref), [`derivative!`](@ref) and [`derivative!!`](@ref).

In addition to [`UnivariateObjective`](@ref), `SimpleSolvers` also contains a [`TemporaryUnivariateObjective`](@ref):
```@example objective
t_obj = TemporaryUnivariateObjective(obj.F, obj.D)
```

!!! info "Why are there two types of univariate objectives?"
    There are two types of univariate objectives in `SimpleSolvers`: `UnivariateObjective`s and `TemporaryUnivariateObjective`s. The latter is only used for allocating line search objectives and contains less functionality.

### Multivariate Objectives

[`MultivariateObjective`](@ref)s are used in a way similar to [`UnivariateObjective`](@ref)s, the difference is that the *derivative functions* are replaced by *gradient functions*, i.e.:
- [`derivative`](@ref) ``\implies`` [`gradient`](@ref),
- [`derivative!`](@ref) ``\implies`` [`gradient!`](@ref),
- [`derivative!!`](@ref) ``\implies`` [`gradient!!`](@ref).

```@example objective
f(x::AbstractArray) = sum(tanh.(x))
x = rand(3)

obj = MultivariateObjective(f, x)
```