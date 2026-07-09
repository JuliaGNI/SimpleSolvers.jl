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
    DogLeg()

*Powell's dogleg method* [powell1970new](@cite).
"""
struct DogLeg <: NonlinearSolverMethod end
