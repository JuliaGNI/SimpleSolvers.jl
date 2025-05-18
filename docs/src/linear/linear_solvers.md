# Linear Solvers

Objects of type [`LinearSolver`](@ref) are used to solve [`LinearSystem`](@ref)s, i.e. we want to find ``x`` for given ``A`` and ``y`` such that

```math
    Ax = y
```

is satisfied. 

A linear system can be called with:

```@example linear_system
using SimpleSolvers

A = [(0. + 1e-5) 1. 2.; 3. 4. 5.; 6. 7. 8.]
y = [1., 2., 3.]
ls = LinearSystem(copy(A), y)
nothing # hide
```

Note that we here use the matrix:

```math
A = \begin{pmatrix} 0 + \varepsilon & 1 & 2 \\ 3 & 4 & 5 \\ 6 & 7 & 8 \end{pmatrix}.
```

This matrix would be singular if we had ``\varepsilon = 0`` because ``2\cdot\begin{pmatrix} 3 \\ 4 \\ 5 \end{pmatrix} - \begin{pmatrix} \end 6 \\ 7 \\ 8 {pmatrix} = \begin{pmatrix} 0 \\ 1 \\ 2 \end{pmatrix}.`` So by choosing ``\varepsilon = 10^{-5}`` the matrix is *ill-conditioned*.

We will also use this problem as an example to demonstrate features of [`LinearSolver`](@ref)s.

As a result we can solve the system in double precision with a naive matrix inversion:

```@example linear_system
inv(A) * y
```

but not in single precision[^1]:
[^1]: We use the superscript ``s`` to indicate single precision.

```@example linear_system
Aˢ = Float32.(A)
yˢ = Float32.(y)
lsˢ = LinearSystem(copy(Aˢ), yˢ)

inv(Aˢ) * yˢ
```

We now use an [`LUSolver`](@ref) to solve the same problem:

```@example linear_system
lu = LUSolver(lsˢ)
solution(lsˢ)
```

If we however deactivate pivoting we get:

```@example linear_system
Aˢ = Float32.(A) # hide
yˢ = Float32.(y) # hide
lsˢ = LinearSystem(copy(Aˢ), yˢ) # hide
lu = LUSolver(lsˢ; pivot = false)
solution(lsˢ)
```