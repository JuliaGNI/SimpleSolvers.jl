# Linear Solvers

Objects of type [`LinearSolver`](@ref) are used to solve [`LinearSystem`](@ref)s, i.e. we want to find ``x`` for given ``A`` and ``y`` such that

```math
    Ax = y
```

is satisfied. 

A linear system can be called with:

```@example linear_system
using SimpleSolvers

A = [1. 2. 3.; 4. 5. 6.; 7. 8. 9.]
y = [1., 1., 1.]
ls = LinearSystem(A, y)
nothing # hide
```

We will also use this problem as an example to demonstrate features of [`LinearSolver`](@ref)s:

We further note that this matrix is ill-conditioned:

```@example linear_system
using LinearAlgebra # hide
det(A)
```

As a result we can solve the system in double precision with a naive matrix inversion:

```@example linear_system
inv(A) * y
```

but not in single precision[^1]:
[^1]: We use the superscript ``s`` to indicate single precision.

```@example linear_system
Aˢ = Float32.(A)
yˢ = Float32.(y)
lsˢ = LinearSystem(Aˢ, yˢ)

inv(Aˢ) * yˢ
```

We now use an [`LUSolver`](@ref) to solve the same problem:

```@example linear_system
lu = LUSolver(lsˢ)
solution(lsˢ)
```