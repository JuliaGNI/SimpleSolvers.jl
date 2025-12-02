
const SOLUTION_MAX_PRINT_LENGTH = 10

"""
    Optimizer

The optimizer that stores all the information needed for an optimization problem. This problem can be solved by calling [`solve!(::AbstractVector, ::Optimizer)`](@ref).

# Keys
- `algorithm::`[`OptimizerState`](@ref),
- `problem::`[`OptimizerProblem`](@ref),
- `config::`[`Options`](@ref),
- `state::`[`OptimizerState`](@ref).
"""
mutable struct Optimizer{T,
                 ALG <: OptimizerMethod,
                 OBJ <: OptimizerProblem{T},
                 GT <: Gradient{T},
                 HT <: Hessian{T},
                 OCT <: OptimizerCache,
                 LST <: LinesearchState} <: AbstractSolver
    algorithm::ALG
    problem::OBJ
    gradient::GT
    hessian::HT
    config::Options{T}
    cache::OCT
    linesearch::LST
    iterations::Int

    function Optimizer(algorithm::OptimizerMethod, problem::OptimizerProblem{T}, hessian::Hessian{T}, cache::OptimizerCache, lst::LinesearchState; gradient = GradientAutodiff{T}(problem.F, length(cache.x)), iterations = 1, options_kwargs...) where {T}
        config = Options(T; options_kwargs...)
        new{T, typeof(algorithm), typeof(problem), typeof(gradient), typeof(hessian), typeof(cache), typeof(lst)}(algorithm, problem, gradient, hessian, config, cache, lst, iterations)
    end
end

function Optimizer(x::VT, problem::OptimizerProblem; algorithm::OptimizerMethod = BFGS(), linesearch::LinesearchMethod = Backtracking(), options_kwargs...) where {T, VT <: AbstractVector{T}}
    cache = NewtonOptimizerCache(x)
    hes = Hessian(algorithm, problem, x)
    Optimizer(algorithm, problem, hes, cache, LinesearchState(linesearch; T = T); options_kwargs...)
end

function Optimizer(x::AbstractVector, F::Function; ∇F! = nothing, mode = :autodiff, kwargs...)
    G = if (ismissing(∇F!)|isnothing(∇F!))
            if mode == :autodiff
                GradientAutodiff(F, x)
            else
                GradientFiniteDifferences(F, x)
            end
        else
            GradientFunction(F, ∇F!, x)
        end
    problem = (ismissing(∇F!)|isnothing(∇F!)) ? OptimizerProblem(F, x) : OptimizerProblem(F, ∇F!, x)
    Optimizer(x, problem; gradient = G, kwargs...)
end

config(opt::Optimizer) = opt.config
problem(opt::Optimizer) = opt.problem
algorithm(opt::Optimizer) = opt.algorithm
linesearch(opt::Optimizer) = opt.linesearch
hessian(opt::Optimizer) = opt.hessian
direction(opt::Optimizer) = direction(cache(opt))
rhs(opt::Optimizer) = rhs(cache(opt))
cache(opt::Optimizer) = opt.cache
iteration_number(opt::Optimizer) = opt.iterations
gradient(opt::Optimizer) = opt.gradient

function increase_iteration_number!(opt)
    opt.iterations = iteration_number(opt) + 1
end

# function Base.show(io::IO, opt::Optimizer)
#     c = config(opt)
#     s = status(opt)
# 
#     @printf io "\n"
#     @printf io " * Algorithm: %s \n" algorithm(opt)
#     @printf io "\n"
#     @printf io " * Linesearch: %s\n" linesearch(opt)
#     @printf io "\n"
#     @printf io " * Iterations\n"
#     @printf io "\n"
#     @printf io "    n = %i\n" iterations(s)
#     @printf io "\n"
#     @printf io " * Convergence measures\n"
#     @printf io "\n"
#     @printf io "    |x - x'|               = %.2e %s %.1e\n"  x_abschange(s) x_abschange(s) ≤ x_abstol(c) ? "≤" : "≰" x_abstol(c)
#     @printf io "    |x - x'|/|x'|          = %.2e %s %.1e\n"  x_relchange(s) x_relchange(s) ≤ x_reltol(c) ? "≤" : "≰" x_reltol(c)
#     @printf io "    |f(x) - f(x')|         = %.2e %s %.1e\n"  f_abschange(s) f_abschange(s) ≤ f_abstol(c) ? "≤" : "≰" f_abstol(c)
#     @printf io "    |f(x) - f(x')|/|f(x')| = %.2e %s %.1e\n"  f_relchange(s) f_relchange(s) ≤ f_reltol(c) ? "≤" : "≰" f_reltol(c)
#     @printf io "    |g(x)|                 = %.2e %s %.1e\n"  g_residual(s)  g_residual(s)  ≤ g_restol(c) ? "≤" : "≰" g_restol(c)
#     @printf io "\n"
# 
# end

check_gradient(opt::Optimizer) = check_gradient(gradient(problem(opt)))
print_gradient(opt::Optimizer) = print_gradient(gradient(problem(opt)))

meets_stopping_criteria(status::OptimizerStatus, opt::Optimizer) = meets_stopping_criteria(status, config(opt), iteration_number(opt))

function initialize!(opt::Optimizer, x::AbstractVector)
    initialize!(problem(opt), x)
    initialize!(cache(opt), x)
    opt.iterations = 0

    opt
end

"""
    update!(opt, x)

Compute problem and gradient at new solution.

This first calls [`update!(::OptimizerResult, ::AbstractVector, ::AbstractVector, ::AbstractVector)`](@ref) and then [`update!(::NewtonOptimizerState, ::AbstractVector)`](@ref).
We note that the [`OptimizerStatus`](@ref) (unlike the [`NewtonOptimizerState`](@ref)) is updated when calling [`update!(::OptimizerResult, ::AbstractVector, ::AbstractVector, ::AbstractVector)`](@ref).
"""
function update!(opt::Optimizer, state::OptimizerState, x::AbstractVector)
    update!(problem(opt), gradient(opt), x)
    update!(cache(opt), state, gradient(opt), hessian(opt), x)

    opt
end

"""
    solver_step!(x, state)

Compute a full iterate for an instance of [`NewtonOptimizerState`](@ref) `state`.

This also performs a line search.

# Examples

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: solver_step!, NewtonOptimizerState)
f(x) = sum(x .^ 2 + x .^ 3 / 3)
x = [1f0, 2f0]
opt = Optimizer(x, f; algorithm = Newton())
state = NewtonOptimizerState(x)

solver_step!(opt, state, x)

# output

2-element Vector{Float32}:
 0.25
 0.6666666
```
"""
function solver_step!(opt::Optimizer, state::OptimizerState, x::VT) where {VT <: AbstractVector}
    # update problem, hessian, state and status
    update!(opt, state, x)

    # solve H δx = - ∇f
    # rhs is -g
    direction(opt) .= hessian(cache(opt)) \ rhs(opt)

    # apply line search
    α = linesearch(opt)(linesearch_problem(problem(opt), gradient(opt), cache(opt), state))

    # compute new minimizer
    x .= compute_new_iterate(x, α, direction(opt))
    cache(opt).x .= x
    x
end

"""
    solve!(x, state, opt)

Solve the optimization problem described by `opt::`[`Optimizer`](@ref) and store the result in `x`.

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: solve!, NewtonOptimizerState, update!; using Random: seed!; seed!(123))
f(x) = sum(x .^ 2 + x .^ 3 / 3)
x = [1f0, 2f0]
opt = Optimizer(x, f; algorithm = Newton())
state = NewtonOptimizerState(x)

solve!(opt, state, x)

# output

SimpleSolvers.OptimizerResult{Float32, Float32, Vector{Float32}, SimpleSolvers.OptimizerStatus{Float32, Float32}}(
 * Convergence measures

    |x - x'|               = 7.82e-03
    |x - x'|/|x'|          = 2.56e+02
    |f(x) - f(x')|         = 9.31e-10
    |f(x) - f(x')|/|f(x')| = 1.00e+00
    |g(x) - g(x')|         = 1.57e-02
    |g(x)|                 = 6.10e-05

, Float32[4.6478817f-8, 3.0517578f-5], 9.313341f-10)
```

We can also check how many iterations it took:

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: solve!, NewtonOptimizerState, update!, iteration_number; using Random: seed!; seed!(123); f(x) = sum(x .^ 2 + x .^ 3 / 3); x = [1f0, 2f0]; opt = Optimizer(x, f; algorithm = Newton()); state = NewtonOptimizerState(x); solve!(opt, state, x))
iteration_number(opt)

# output

4
```
Too see the value of `x` after one iteration confer the docstring of [`solver_step!`](@ref).
"""
function solve!(opt::Optimizer, state::OptimizerState, x::AbstractVector)
    initialize!(opt, x)

    while true
        increase_iteration_number!(opt)
        solver_step!(opt, state, x)
        status = OptimizerStatus(state, cache(opt), value(problem(opt)); config = config(opt))
        meets_stopping_criteria(status, opt) && break
        update!(state, problem(opt), gradient(opt), x)
    end

    warn_iteration_number(opt, config(opt))

    status = OptimizerStatus(state, cache(opt), value(problem(opt)); config = config(opt))
    OptimizerResult(status, x, value(problem(opt)))
end

function warn_iteration_number(opt::Optimizer, config::Options)
    if config.warn_iterations > 0 && iteration_number(opt) ≥ config.warn_iterations
        println("WARNING: Optimizer took ", iteration_number(opt), " iterations.")
    end
end