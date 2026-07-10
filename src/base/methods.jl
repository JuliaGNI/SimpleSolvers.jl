"""
    NonlinearSolverMethod <: SolverMethod

A supertype collecting all nonlinear *solver* methods, i.e. [`Newton`](@ref),
[`Picard`](@ref) and [`DogLeg`](@ref).

Compare this with [`LinesearchMethod`](@ref): both are subtypes of `SolverMethod`,
but a `LinesearchMethod` describes a one-dimensional line search (used *inside* a
solver step) whereas a `NonlinearSolverMethod` describes the outer nonlinear
iteration itself.
"""
abstract type NonlinearSolverMethod <: SolverMethod end

"""
    Newton <: NonlinearSolverMethod

# Constructors

```jldoctest; setup = :(using SimpleSolvers)
Newton()

# output

Newton{true}(1)
```

```jldoctest; setup = :(using SimpleSolvers)
QuasiNewton()

# output

QuasiNewton(5)
```
!!! info
    The *refactorize* parameter determines how often the Jacobian is refactored. This is the difference between the [`NewtonSolver`](@ref) and [`QuasiNewtonSolver`](@ref).
"""
struct Newton{QT} <: NonlinearSolverMethod
    refactorize::Int

    Newton{true}(refactorize::Integer=1) = new{true}(refactorize)
    Newton{false}(refactorize::Integer=DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER) = new{false}(refactorize)
end

Newton() = Newton{true}()

"""
The default number of iterations before the [`Jacobian`](@ref) is refactored in the [`QuasiNewtonSolver`](@ref)
"""
const DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER = 5

const QuasiNewton = Newton{false}

"""
    Picard <: NonlinearSolverMethod

See [`PicardSolver`](@ref).
"""
struct Picard <: NonlinearSolverMethod end

"""
    DogLeg(refactorize=1)

*Powell's dogleg method* [powell1970new](@cite).

Like [`Newton`](@ref), the `refactorize` parameter determines after how many
steps the [`Jacobian`](@ref) is re-evaluated and refactored (see [`factorize!`](@ref)).
The default `refactorize = 1` re-evaluates and refactorizes the Jacobian on every step;
`refactorize > 1` reuses the Jacobian (and its factorization) in between, giving a
quasi-Newton-style dogleg method.
"""
struct DogLeg <: NonlinearSolverMethod
    refactorize::Int

    DogLeg(refactorize::Integer=1) = new(refactorize)
end
