using SimpleSolvers
using SimpleSolvers: NonlinearSolverException
using Test

function F(y::AbstractVector{T}, x::AbstractVector{T}, params) where {T}
    y .= exp.(-one(T) ./ (x .^ 2))
end

const n = 10
const T = Float32
J₁ = JacobianFiniteDifferences{T}(F, n, n) # the finite difference Jacobian doesn't return NaNs in the first iteration.
J₂ = JacobianAutodiff{T}(F, n)
x = zeros(T, n)
y = zeros(T, n)

nl₁ = NonlinearSolver(NewtonMethod(), x, y; F=F, jacobian=J₁, verbosity=2)
nl₂ = NonlinearSolver(NewtonMethod(), x, y; F=F, jacobian=J₂, verbosity=2)

x₁ = zeros(T, n)
x₂ = zeros(T, n)
@test_throws NonlinearSolverException solve!(x₁, nl₁)
@test_throws NonlinearSolverException solve!(x₂, nl₂)
