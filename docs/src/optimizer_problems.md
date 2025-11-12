# Optimizer Problems

A central object in `SimpleSolvers` are *optimizer problems* (see [`SimpleSolvers.AbstractOptimizerProblem`](@ref)). They are either [`SimpleSolvers.AbstractUnivariateProblem`](@ref)s or [`MultivariateOptimizerProblem`](@ref)s. The goal of a *solver* (both [`LinearSolver`](@ref)s and [`NonlinearSolver`](@ref)s) is to make the optimizer problem have value zero. The goal of an [`Optimizer`](@ref) is to minimize a [`MultivariateOptimizerProblem`](@ref).

## Examples

### Univariate Optimizer Problems

We can allocate a univariate optimizer problem with:

```@example optimizer_problem
using SimpleSolvers
using Random; Random.seed!(123) # hide

f(x::Number) = x ^ 2
x = rand()
obj = UnivariateProblem(f, x)
```

Associated to [`UnivariateProblem`](@ref) are the following functions (amongst others):
- [`value`](@ref)
- [`value!`](@ref)
- [`value!!`](@ref)
- [`derivative`](@ref)
- [`derivative!`](@ref)
- [`derivative!!`](@ref)

The function [`value`](@ref) evaluates the optimizer problem at the provided input and increases the counter by 1:

```@example optimizer_problem
y = value(obj, x)
```

We can check how the function call changed `obj`:

```@example optimizer_problem
obj
```

The stored value for `f` has not been updated. In order to so we can call the in-place function [`value!`](@ref):

```@example optimizer_problem
y = value!(obj, x)
obj
```

We further note that `SimpleSolvers` contains another function [`value!!`](@ref) that *forces evaluation*. This is opposed to [`value!`](@ref) which does not always force evaluation:

```@example optimizer_problem
y = value!(obj, x)
obj
```

So [`value!`](@ref) first checks if the optimizer problem `obj` has already been called on `x`. In order to force another evaluation we can write:

```@example optimizer_problem
y = value!!(obj, x)
obj
```

The function [`value`](@ref) can also be called without additional input arguments:

```@example optimizer_problem
value(obj)
```

But then the associated function is not called again (calling [`value`](@ref) this way does not increase the counter):

```@example optimizer_problem
obj
```

An equivalent relationship exists between the functions [`derivative`](@ref), [`derivative!`](@ref) and [`derivative!!`](@ref).

In addition to [`UnivariateProblem`](@ref), `SimpleSolvers` also contains a [`LinesearchProblem`](@ref)[^1]:

[^1]: To be used together with [`SimpleSolvers.linesearch_problem`](@ref).

```@example optimizer_problem
t_obj = LinesearchProblem(obj.F, obj.D)
```

!!! info "Why are there two types of univariate optimizer problems?"
    There are two types of univariate optimizer problems in `SimpleSolvers`: `UnivariateProblem`s and `LinesearchProblem`s. The latter is only used for allocating line search problems and contains less functionality.

### Multivariate Optimizer Problems

[`MultivariateOptimizerProblem`](@ref)s are used in a way similar to [`UnivariateProblem`](@ref)s, the difference is that the *derivative functions* are replaced by *gradient functions*, i.e.:
- [`derivative`](@ref) ``\implies`` [`gradient`](@ref),
- [`derivative!`](@ref) ``\implies`` [`gradient!`](@ref),
- [`derivative!!`](@ref) ``\implies`` [`gradient!!`](@ref).

```@example optimizer_problem
Random.seed!(123) # hide
f(x::AbstractArray) = sum(x .^ 2)
x = rand(3)

obj = MultivariateOptimizerProblem(f, x)
```

Every instance of [`MultivariateOptimizerProblem`](@ref) stores an instance of [`Gradient`](@ref) to which we [similarly can apply the functions](@ref "Gradients") [`gradient`](@ref) or [`gradient!`](@ref):

```@example optimizer_problem
gradient(obj, x)
```

The difference to [`Gradient`](@ref) is that we also store the value for the evaluated gradient, which can be accessed by calling:

```@example optimizer_problem
gradient(obj)
```