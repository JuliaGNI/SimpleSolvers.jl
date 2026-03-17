abstract type NonlinearMethod <: SolverMethod end

"A supertype collecting all nonlinear methods, including [`NewtonMethod`](@ref)s, [`PicardMethod`](@ref) and [`DogLeg`](@ref)."
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

    NewtonMethod() = new{true}(1)

    NewtonMethod{false}(refactorize=DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER) = new{false}(refactorize)
end

"""
The default number of iterations before the [`Jacobian`](@ref) is refactored in the [`QuasiNewtonSolver`](@ref)
"""
const DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER = 5

const QuasiNewtonMethod = NewtonMethod{false}

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
