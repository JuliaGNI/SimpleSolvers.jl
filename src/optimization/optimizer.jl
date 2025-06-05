
const SOLUTION_MAX_PRINT_LENGTH = 10

"""
An `OptimizationAlgorithm` is a data structure that is used to dispatch on different algorithms.

It needs to implement three methods,
```
initialize!(alg::OptimizationAlgorithm, ::AbstractVector)
update!(alg::OptimizationAlgorithm, ::AbstractVector)
solver_step!(::AbstractVector, alg::OptimizationAlgorithm)
```
that initialize and update the state of the algorithm and perform an actual optimization step.

Further the following convenience methods should be implemented,
```
objective(alg::OptimizationAlgorithm)
gradient(alg::OptimizationAlgorithm)
hessian(alg::OptimizationAlgorithm)
linesearch(alg::OptimizationAlgorithm)
```
which return the objective to optimize, its gradient and (approximate) Hessian as well as the
linesearch algorithm used in conjunction with the optimization algorithm if any.

See [`NewtonOptimizerState`](@ref) for a `struct` that was derived from `OptimizationAlgorithm`.

!!! info
    Note that a `OptimizationAlgorithm` is not necessarily a `NewtonOptimizerState` as we can also have other optimizers, *Adam* for example.
"""
abstract type OptimizationAlgorithm end

OptimizerState(alg::OptimizationAlgorithm, args...; kwargs...) = error("OptimizerState not implemented for $(typeof(alg))")

"""
    isaOptimizationAlgorithm(alg)

Verify if an object implements the [`OptimizationAlgorithm`](@ref) interface.
"""
function isaOptimizationAlgorithm(alg)
    x = rand(3)

    applicable(gradient, alg) &&
    applicable(hessian, alg) &&
    applicable(linesearch, alg) &&
    applicable(objective, alg) &&
    applicable(initialize!, alg, x) &&
    applicable(update!, alg, x) &&
    applicable(solver_step!, x, alg)
end

"""
    Optimizer

The optimizer that stores all the information needed for an optimization problem. This problem can be solved by calling [`solve!(::AbstractVector, ::Optimizer)`](@ref).

# Keys
- `algorithm::`[`OptimizationAlgorithm`](@ref),
- `objective::`[`MultivariateObjective`](@ref),
- `config::`[`Options`](@ref),
- `result::`[`OptimizerResult`](@ref),
- `state::`[`OptimizationAlgorithm`](@ref).
"""
struct Optimizer{T,
                 ALG <: NonlinearMethod,
                 OBJ <: MultivariateObjective{T},
                 HT <: Hessian{T},
                 RES <: OptimizerResult{T},
                 AST <: OptimizationAlgorithm} <: NonlinearSolver
    algorithm::ALG
    objective::OBJ
    hessian::HT
    config::Options{T}
    result::RES
    state::AST

    function Optimizer(algorithm::NonlinearMethod, objective::MultivariateObjective{T}, hessian::Hessian{T}, result::OptimizerResult{T}, state::OptimizationAlgorithm; options_kwargs...) where {T}
        config = Options(T; options_kwargs...)
        new{T, typeof(algorithm), typeof(objective), typeof(hessian), typeof(result), typeof(state)}(algorithm, objective, hessian, config, result, state)
    end
end

function Optimizer(x::VT, objective::MultivariateObjective; algorithm::NewtonMethod = BFGS(), linesearch::LinesearchMethod = Backtracking(), options_kwargs...) where {T, VT <: AbstractVector{T}}
    y = value(objective, x)
    result = OptimizerResult(x, y)
    clear!(result)
    astate = NewtonOptimizerState(x; linesearch = linesearch)
    hes = Hessian(algorithm, objective, x)
    Optimizer(algorithm, objective, hes, result, astate; options_kwargs...)
end

function Optimizer(x::AbstractVector, F::Function; ∇F! = nothing, kwargs...)
    G = Gradient(∇F!, F, x)
    objective = MultivariateObjective(F, G, x)
    Optimizer(x, objective; kwargs...)
end

config(opt::Optimizer) = opt.config
result(opt::Optimizer) = opt.result
status(opt::Optimizer) = opt.result.status
state(opt::Optimizer) = opt.state
objective(opt::Optimizer) = opt.objective
algorithm(opt::Optimizer) = opt.algorithm
linesearch(opt::Optimizer) = linesearch(state(opt))
hessian(opt::Optimizer) = opt.hessian
direction(opt::Optimizer) = direction(state(opt))
rhs(opt::Optimizer) = rhs(state(opt))
cache(opt::Optimizer) = cache(state(opt))
iteration_number(opt::Optimizer) = iteration_number(status(opt))
gradient(::Optimizer) = error("There is an ambiguity in calling gradient on Optimizer at the moment, as the cache, the result and the objective all store this information.")

Base.minimum(opt::Optimizer) = minimum(result(opt))
minimizer(opt::Optimizer) = minimizer(result(opt))

function Base.show(io::IO, opt::Optimizer)
    c = config(opt)
    s = status(opt)

    @printf io "\n"
    @printf io " * Algorithm: %s \n" algorithm(opt)
    @printf io "\n"
    @printf io " * Linesearch: %s\n" linesearch(opt)
    @printf io "\n"
    @printf io " * Iterations\n"
    @printf io "\n"
    @printf io "    n = %i\n" iterations(s)
    @printf io "\n"
    @printf io " * Convergence measures\n"
    @printf io "\n"
    @printf io "    |x - x'|               = %.2e %s %.1e\n"  x_abschange(s) x_abschange(s) ≤ x_abstol(c) ? "≤" : "≰" x_abstol(c)
    @printf io "    |x - x'|/|x'|          = %.2e %s %.1e\n"  x_relchange(s) x_relchange(s) ≤ x_reltol(c) ? "≤" : "≰" x_reltol(c)
    @printf io "    |f(x) - f(x')|         = %.2e %s %.1e\n"  f_abschange(s) f_abschange(s) ≤ f_abstol(c) ? "≤" : "≰" f_abstol(c)
    @printf io "    |f(x) - f(x')|/|f(x')| = %.2e %s %.1e\n"  f_relchange(s) f_relchange(s) ≤ f_reltol(c) ? "≤" : "≰" f_reltol(c)
    @printf io "    |g(x)|                 = %.2e %s %.1e\n"  g_residual(s)  g_residual(s)  ≤ g_restol(c) ? "≤" : "≰" g_restol(c)
    @printf io "\n"

    @printf io " * Candidate solution\n"
    @printf io "\n"
    length(minimizer(opt)) > SOLUTION_MAX_PRINT_LENGTH || @printf io  "    Final solution value:     [%s]\n" join([@sprintf "%e" x for x in minimizer(opt)], ", ")
    @printf io "    Final objective value:     %e\n" minimum(opt)
    @printf io "\n"

end

check_gradient(opt::Optimizer) = check_gradient(gradient(objective(opt)))
print_gradient(opt::Optimizer) = print_gradient(gradient(objective(opt)))
print_status(opt::Optimizer) = print_status(status(opt), config(opt))

assess_convergence(opt::Optimizer) = assess_convergence(status(opt), config(opt))
meets_stopping_criteria(opt::Optimizer) = meets_stopping_criteria(status(opt), config(opt))

function initialize!(opt::Optimizer, x::AbstractVector)
    initialize!(objective(opt), x)
    initialize!(result(opt), x)
    initialize!(state(opt), x)
    initialize!(hessian(opt), x)

    opt
end

"""
    update!(opt, x)

Compute objective and gradient at new solution and update result.

This first calls [`update!(::OptimizerResult, ::AbstractVector, ::AbstractVector, ::AbstractVector)`](@ref) and then [`update!(::NewtonOptimizerState, ::AbstractVector)`](@ref).
We note that the [`OptimizerStatus`](@ref) (unlike the [`NewtonOptimizerState`](@ref)) is updated when calling [`update!(::OptimizerResult, ::AbstractVector, ::AbstractVector, ::AbstractVector)`](@ref).
"""
function update!(opt::Optimizer, x::AbstractVector)
    update!(objective(opt), x)
    update!(hessian(opt), x)
    update!(state(opt), x, gradient(objective(opt)), hessian(opt))
    update!(result(opt), x, value(objective(opt)), gradient(objective(opt)))

    opt
end

"""
    solver_step!(x, state)

Compute a full iterate for an instance of [`NewtonOptimizerState`](@ref) `state`.

This also performs a line search.

# Examples

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: solver_step!, NewtonOptimizerState, update!)
f(x) = sum(x .^ 2 + x .^ 3 / 3)
x = [1f0, 2f0]
opt = Optimizer(x, f; algorithm = Newton())

solver_step!(opt, x)

# output

2-element Vector{Float32}:
 0.25
 0.6666666
```
"""
function solver_step!(opt::Optimizer, x::VT)::VT where {VT <: AbstractVector}
    # update objective, hessian, state and result
    update!(opt, x)

    # solve H δx = - ∇f
    # rhs is -g
    ldiv!(direction(opt), hessian(opt), rhs(opt))

    # apply line search
    α = linesearch(state(opt))(linesearch_objective(objective(opt), cache(opt)))

    # compute new minimizer
    x .= compute_new_iterate(x, α, direction(opt))
end

"""
    solve!(x, opt)

Solve the optimization problem described by `opt::`[`Optimizer`](@ref) and store the result in `x`.

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: solve!, NewtonOptimizerState, update!)
f(x) = sum(x .^ 2 + x .^ 3 / 3)
x = [1f0, 2f0]
opt = Optimizer(x, f; algorithm = Newton())

solve!(opt, x)

# output
2-element Vector{Float32}:
 0.0
 4.656613f-10
```

We can also check how many iterations it took:

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: solve!, NewtonOptimizerState, update!, iteration_number; f(x) = sum(x .^ 2 + x .^ 3 / 3); x = [1f0, 2f0]; opt = Optimizer(x, f; algorithm = Newton()); solve!(opt, x))
iteration_number(opt)

# output

15
```
Too see the value of `x` after one iteration confer the docstring of [`solver_step!`](@ref).
"""
function solve!(opt::Optimizer, x::AbstractVector)
    initialize!(opt, x)

    while (iteration_number(opt) == 0 || !meets_stopping_criteria(opt))
        increase_iteration_number!(result(opt))
        solver_step!(opt, x)
        update!(opt, x)
    end

    warn_iteration_number(status(opt), config(opt))
    print_status(status(opt), config(opt))

    x
end