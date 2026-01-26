"""
    NewtonSolver

A `const` derived from [`NonlinearSolver`](@ref)

# Constructors

The `NewtonSolver` can be called with an [`NonlinearProblem`](@ref) or with a `Callable`. Note however that the latter will probably be deprecated in the future.
```jldoctest; setup=:(using SimpleSolvers)
linesearch = Quadratic()
F(y, x, params) = y .= tanh.(x)
x = [.5, .5]
y = zero(x)
F(y, x, nothing)

NewtonSolver(x, y; F = F, linesearch = linesearch)

# output

i=   0,
x= NaN,
f= NaN,
rxₐ= NaN,
rfₐ= NaN
```

What is shown here is the status of the `NewtonSolver`, i.e. an instance of [`NonlinearSolverStatus`](@ref).

# Keywords
- `nonlinearproblem::`[`NonlinearProblem`](@ref): the system that has to be solved. This can be accessed by calling [`nonlinearproblem`](@ref),
- `jacobian::`[`Jacobian`](@ref)
- `linear::`[`LinearSolver`](@ref): the linear solver is used to compute the [`direction`](@ref) of the solver step (see [`solver_step!`](@ref)). This can be accessed by calling [`linearsolver`](@ref),
- `linesearch::`[`LinesearchState`](@ref)
- `refactorize::Int`: determines after how many steps the Jacobian is updated and refactored (see [`factorize!`](@ref)). If we have `refactorize > 1`, then we speak of a [`QuasiNewtonSolver`](@ref),
- `cache::`[`NonlinearSolverCache`](@ref)
- `config::`[`Options`](@ref)
- `status::`[`NonlinearSolverStatus`](@ref):
"""
const NewtonSolver{T} = NonlinearSolver{T,NewtonMethod}

function NewtonSolver(x::AT, nlp::NLST, ls::LST, linearsolver::LSoT, linesearch::LiSeT, cache::CT; jacobian::Jacobian=JacobianAutodiff(nlp.F, x), refactorize::Integer=1, options_kwargs...) where {T,AT<:AbstractVector{T},NLST,LST,LSoT,LiSeT,CT}
    cache = NonlinearSolverCache(x, x)
    NonlinearSolver(x, nlp, ls, linearsolver, linesearch, cache; method=NewtonMethod(refactorize), jacobian=jacobian, options_kwargs...)
end

function QuasiNewtonSolver(x, nlp, ls, linearsolver, linesearch, cache; options_kwargs...)
    NewtonSolver(x, nlp, ls, linearsolver, linesearch, cache; refactorize=DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER, options_kwargs...)
end

"""
    NewtonSolver(x, F, y)

# Keywords
- `linear_solver_method`
- `DF!`
- `linesearch`
- `mode`
- `options_kwargs`: see [`Options`](@ref)
"""
function NewtonSolver(x::AT, F::Callable, y::AT; linear_solver_method=LU(), (DF!)=missing, linesearch=Backtracking(), jacobian=JacobianAutodiff(F, x, y), kwargs...) where {T,AT<:AbstractVector{T}}
    nlp = ismissing(DF!) ? NonlinearProblem(F, x, y) : NonlinearProblem(F, DF!, x, y)
    jacobian = ismissing(DF!) ? jacobian : JacobianFunction{T}(F, DF!)
    cache = NonlinearSolverCache(x, y)
    linearproblem = LinearProblem(alloc_j(x, y))
    linearsolver = LinearSolver(linear_solver_method, y)
    ls = LinesearchState(linesearch; T=T)
    NewtonSolver(x, nlp, linearproblem, linearsolver, ls, cache; jacobian=jacobian, kwargs...)
end

function NewtonSolver(x::AT, y::AT; F=missing, kwargs...) where {T,AT<:AbstractVector{T}}
    !ismissing(F) || error("You have to provide an F.")
    NewtonSolver(x, F, y; kwargs...)
end

function direction!(d::AbstractVector{T}, x::AbstractVector{T}, s::NewtonSolver{T}, params::OptionalParameters) where {T} 
     # first we update the rhs of the linearproblem 
     value!(rhs(linearproblem(s)), nonlinearproblem(s), x, params) 
     rhs(linearproblem(s)) .*= -1
     # for a quasi-Newton method the Jacobian isn't updated in every iteration 
     if (mod(iteration_number(s) - 1, method(s).refactorize) == 0 || iteration_number(s) == 1) 
         jacobian!(s, x, params) 
         update!(linearproblem(s), jacobian(s)) 
         factorize!(linearsolver(s), linearproblem(s)) 
     end 
     ldiv!(d, linearsolver(s), rhs(linearproblem(s))) 
end

function direction!(s::NewtonSolver, x::AbstractVector, params::OptionalParameters)
    direction!(direction(cache(s)), x, s, params)
end

"""
    QuasiNewtonSolver

A convenience constructor for [`NewtonSolver`](@ref). Also see [`DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER`](@ref).

Calling `QuasiNewtonSolver` hence produces an instance of [`NewtonSolver`](@ref) with the difference that `refactorize ≠ 1`. The [`Jacobian`](@ref) is thus not evaluated and refactored in every step.

# Implementation
It does:

```julia
QuasiNewtonSolver(args...; kwargs...) = NewtonSolver(args...; refactorize=DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER, kwargs...)
```
"""
QuasiNewtonSolver(args...; kwargs...) = NewtonSolver(args...; refactorize=DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER, kwargs...)


check_jacobian(s::NewtonSolver) = check_jacobian(jacobian(s))
print_jacobian(s::NewtonSolver) = print_jacobian(jacobian(s))


"""
    update!(solver, x, params)

Update the `solver::`[`NewtonSolver`](@ref) based on `x`.
This updates the cache (instance of type [`NonlinearSolverCache`](@ref)) and the status (instance of type [`NonlinearSolverStatus`](@ref)). In course of updating the latter, we also update the `nonlinear` stored in `solver` (and `status(solver)`).

!!! info
    At the moment this is neither used in `solver_step!` nor `solve!`.
"""
function update!(s::NewtonSolver, x₀::AbstractArray, params)
    update!(status(s), x₀, nonlinearproblem(s), params)
    # update!(nonlinearproblem(s), Jacobian(s), x₀, params)
    update!(cache(s), x₀)

    s
end
