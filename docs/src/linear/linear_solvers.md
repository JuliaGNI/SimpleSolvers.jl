# Linear Solvers

Objects of type [`LinearSolver`](@ref) are used to solve [`LinearSystem`](@ref)s, i.e. we want to find ``x`` for given ``A`` and ``y`` such that

```math
    Ax = y
```

is satisfied. 

A linear system can be called with[^1]:

[^1]: Here we also have to *update* the [`LinearSystem`](@ref) by calling [`update!`](@ref). For more information see the [page on the `initialize!` function](@ref "Initialization").

```@example linear_system
using SimpleSolvers

A = [(0. + 1e-6) 1. 2.; 3. 4. 5.; 6. 7. 8.]
y = [1., 2., 3.]
ls = LinearSystem(A, y)
update!(ls, A, y)
nothing # hide
```

Note that we here use the matrix:

```math
A = \begin{pmatrix} 0 + \varepsilon & 1 & 2 \\ 3 & 4 & 5 \\ 6 & 7 & 8 \end{pmatrix}.
```

This matrix would be singular if we had ``\varepsilon = 0`` because ``2\cdot\begin{pmatrix} 3 \\ 4 \\ 5 \end{pmatrix} - \begin{pmatrix} 6 \\ 7 \\ 8 \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \\ 2 \end{pmatrix}.`` So by choosing ``\varepsilon = 10^{-6}`` the matrix is *ill-conditioned*.

We first solve [`LinearSystem`](@ref) with an lu solver (using [`LU`](@ref) and [`solve`](@ref)) in double precision and without pivoting:

```@example linear_system
lu = LU(; pivot = false)
y¹ = solve(lu, ls)
```

We check the result:

```@example linear_system
A * y¹
```

We now do the same in single precision:

```@example linear_system
Aˢ = Float32.(A)
yˢ = Float32.(y)
lsˢ = LinearSystem(Aˢ, yˢ)
update!(lsˢ, Aˢ, yˢ)
y² = solve(lu, lsˢ)
```

and again check the result:

```@example linear_system
Aˢ * y²
```

As we can see the computation of the factorization returns a wrong solution. If we use pivoting however, the problem can also be solved with single precision:

```@example linear_system
lu = LU(; pivot = true)
y³ = solve(lu, lsˢ)
```

```@example linear_system
Aˢ * y³
```

## Solving the System with Built-In Functionality from the `LinearAlgebra` Package

We further try to solve the system with the `inv` operator from the `LinearAlgebra` package. First in double precision:

```@example linear_system
inv(A) * y
```

And also in single precision

```@example linear_system
inv(Aˢ) * yˢ
```

In single precision the result is completely wrong as can also be seen by computing:

```@example linear_system
inv(Aˢ) * Aˢ
```

If we however write:

```@example linear_system
Aˢ \ yˢ
```

we again obtain a correct-looking result, as `LinearAlgebra.\` uses an algorithm very similar to [`factorize!`](@ref) in `SimpleSolvers`.