
const SOLUTION_MAX_PRINT_LENGTH = 10

"""
An `OptimizationAlgorithm` is a datastructe that is used to dispatch on different algorithms.

It needs to implement three important methods,
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
"""
abstract type OptimizationAlgorithm end

OptimizerState(alg::OptimizationAlgorithm, args...; kwargs...) = error("OptimizerState not implemented for $(typeof(alg))")

"""
Verifies if an object implements the [`OptimizationAlgorithm`](@ref) interface.
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


struct Optimizer{ALG <: NonlinearMethod,
                 OBJ <: MultivariateObjective,
                 OPT <: Options,
                 RES <: OptimizerResult,
                 AST <: OptimizationAlgorithm} <: NonlinearSolver
    algorithm::ALG
    objective::OBJ
    config::OPT
    result::RES
    state::AST
end

function Optimizer(x::VT, objective::MultivariateObjective; algorithm = BFGS(), linesearch = Backtracking(), config = Options(XT)) where {XT, VT <: AbstractVector{XT}}
    y = value(objective, x)
    result = OptimizerResult(x, y)
    astate = OptimizerState(algorithm, objective, x, y; linesearch = linesearch)

    Optimizer{typeof(algorithm), typeof(objective), typeof(config), typeof(result), typeof(astate)}(algorithm, objective, config, result, astate)
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
linesearch(opt::Optimizer) = linesearch(opt.state)

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
    clear!(objective(opt))
    initialize!(result(opt), x, value!(objective(opt), x), gradient!(objective(opt), x))
    initialize!(state(opt), x)
end

"compute objective and gradient at new solution and update result"
function update!(opt::Optimizer, x::AbstractVector)
    update!(result(opt), x, value!(objective(opt), x), gradient!(objective(opt), x))
end

function solve!(x, opt::Optimizer)
    initialize!(opt, x) 

    while !meets_stopping_criteria(opt)
        next_iteration!(result(opt))
        solver_step!(x, state(opt))
        update!(opt, x)
    end

    warn_iteration_number(status(opt), config(opt))
    print_status(status(opt), config(opt))

    return x
end
