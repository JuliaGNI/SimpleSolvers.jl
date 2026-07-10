"""
    NonlinearSolverMethod <: SolverMethod

A supertype collecting all nonlinear *solver* methods, i.e. [`NewtonMethod`](@ref),
[`PicardMethod`](@ref) and [`DogLeg`](@ref).

Compare this with [`LinesearchMethod`](@ref): both are subtypes of `SolverMethod`,
but a `LinesearchMethod` describes a one-dimensional line search (used *inside* a
solver step) whereas a `NonlinearSolverMethod` describes the outer nonlinear
iteration itself.
"""
abstract type NonlinearSolverMethod <: SolverMethod end

"""
    NewtonMethod <: NonlinearSolverMethod

# Constructors

```jldoctest; setup = :(using SimpleSolvers)
NewtonMethod()

# output

NewtonMethod{true}(1)
```

```jldoctest; setup = :(using SimpleSolvers)
QuasiNewtonMethod()

# output

QuasiNewtonMethod(5)
```
!!! info
    The *refactorize* parameter determines how often the Jacobian is refactored. This is the difference between the [`NewtonSolver`](@ref) and [`QuasiNewtonSolver`](@ref).
"""
struct NewtonMethod{QT} <: NonlinearSolverMethod
    refactorize::Int

    NewtonMethod{true}(refactorize::Integer=1) = new{true}(refactorize)
    NewtonMethod{false}(refactorize::Integer=DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER) = new{false}(refactorize)
end

NewtonMethod() = NewtonMethod{true}()

"""
The default number of iterations before the [`Jacobian`](@ref) is refactored in the [`QuasiNewtonSolver`](@ref)
"""
const DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER = 5

const QuasiNewtonMethod = NewtonMethod{false}
const Newton = NewtonMethod

"""
    PicardMethod <: NonlinearSolverMethod

See [`PicardSolver`](@ref).
"""
struct PicardMethod <: NonlinearSolverMethod end

"""
    DogLeg(refactorize=1)

*Powell's dogleg method* [powell1970new](@cite).

Like [`NewtonMethod`](@ref), the `refactorize` parameter determines after how many
steps the [`Jacobian`](@ref) is re-evaluated and refactored (see [`factorize!`](@ref)).
The default `refactorize = 1` re-evaluates and refactorizes the Jacobian on every step;
`refactorize > 1` reuses the Jacobian (and its factorization) in between, giving a
quasi-Newton-style dogleg method.
"""
struct DogLeg <: NonlinearSolverMethod
    refactorize::Int

    DogLeg(refactorize::Integer=1) = new(refactorize)
end
