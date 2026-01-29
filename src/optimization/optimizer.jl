
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
struct Optimizer{T,
                 ALG <: OptimizerMethod,
                 OBJ <: OptimizerProblem{T},
                 GT <: Gradient{T},
                 HT <: Hessian{T},
                 OCT <: OptimizerCache,
                 LST <: Linesearch} <: AbstractSolver
    algorithm::ALG
    problem::OBJ
    gradient::GT
    hessian::HT
    config::Options{T}
    cache::OCT
    linesearch::LST

    function Optimizer(algorithm::OptimizerMethod, problem::OptimizerProblem{T}, hessian::Hessian{T}, cache::OptimizerCache, lst::Linesearch; gradient = GradientAutodiff{T}(problem.F, length(cache.x)), options_kwargs...) where {T}
        config = Options(T; options_kwargs...)
        new{T, typeof(algorithm), typeof(problem), typeof(gradient), typeof(hessian), typeof(cache), typeof(lst)}(algorithm, problem, gradient, hessian, config, cache, lst)
    end
end

function Optimizer(x::VT, problem::OptimizerProblem; algorithm::OptimizerMethod = BFGS(), linesearch::LinesearchMethod = Backtracking(), options_kwargs...) where {T, VT <: AbstractVector{T}}
    cache = OptimizerCache(algorithm, x)
    hes = Hessian(algorithm, problem, x)
    Optimizer(algorithm, problem, hes, cache, Linesearch(linesearch; T = T); options_kwargs...)
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
gradient(opt::Optimizer) = opt.gradient

check_gradient(opt::Optimizer) = check_gradient(gradient(problem(opt)))
print_gradient(opt::Optimizer) = print_gradient(gradient(problem(opt)))

meets_stopping_criteria(status::OptimizerStatus, opt::Optimizer, state::OptimizerState) = meets_stopping_criteria(status, config(opt), iteration_number(state))

function initialize!(opt::Optimizer, x::AbstractVector)
    initialize!(cache(opt), x)

    opt
end

"""
    update!(opt, x)

Compute problem and gradient at new solution.

This first calls [`update!(::OptimizerResult, ::AbstractVector, ::AbstractVector, ::AbstractVector)`](@ref) and then [`update!(::NewtonOptimizerState, ::AbstractVector)`](@ref).
We note that the [`OptimizerStatus`](@ref) (unlike the [`NewtonOptimizerState`](@ref)) is updated when calling [`update!(::OptimizerResult, ::AbstractVector, ::AbstractVector, ::AbstractVector)`](@ref).
"""
function update!(opt::Optimizer, state::OptimizerState, x::AbstractVector)
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

solver_step!(x, state, opt)

# output

2-element Vector{Float32}:
 0.25
 0.6666666
```
"""
function solver_step!(x::VT, state::OptimizerState, opt::Optimizer) where {VT <: AbstractVector}
    # update problem, hessian, state and status
    update!(opt, state, x)

    # solve H δx = - ∇f
    # rhs is -g
    compute_direction(opt, state)

    # apply line search
    α = solve(linesearch_problem(problem(opt), gradient(opt), cache(opt), state), linesearch(opt))

    # compute new minimizer
    compute_new_iterate!(x, α, direction(opt))

    # cache has to be updated to compute the correct status
    cache(opt).x .= x
    gradient(opt)(cache(opt).g, x)
    x
end

function compute_direction(opt::Optimizer{T}, ::OptimizerState) where {T}
    direction(opt) .= hessian(cache(opt)) \ rhs(opt)
end

function compute_direction(opt::Optimizer{T, IOM}, state::Union{BFGSState, DFPState}) where {T, IOM <: QuasiNewtonOptimizerMethod}
    direction(opt) .= inverse_hessian(state) * rhs(opt)
end

"""
    solve!(x, state, opt)

Solve the optimization problem described by `opt::`[`Optimizer`](@ref) and store the result in `x`.

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: solve!, NewtonOptimizerState, update!; using Random: seed!; seed!(123))
f(x) = sum(x .^ 2 + x .^ 3 / 3)
x = [1f0, 2f0]
opt = Optimizer(x, f; algorithm = Newton())
state = NewtonOptimizerState(x)

solve!(x, state, opt)

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

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: solve!, NewtonOptimizerState, update!, iteration_number; using Random: seed!; seed!(123); f(x) = sum(x .^ 2 + x .^ 3 / 3); x = [1f0, 2f0]; opt = Optimizer(x, f; algorithm = Newton()); state = NewtonOptimizerState(x); solve!(x, state, opt))
iteration_number(state)

# output

4
```
Too see the value of `x` after one iteration confer the docstring of [`solver_step!`](@ref).
"""
function solve!(x::AbstractVector, state::OptimizerState, opt::Optimizer)
    initialize_state!(state)

    while true
        increase_iteration_number!(state)
        solver_step!(x, state, opt)
        status = OptimizerStatus(state, cache(opt), value(problem(opt), x); config = config(opt))
        meets_stopping_criteria(status, opt, state) && break
        update!(state, gradient(opt), x)
    end

    warn_iteration_number(state, config(opt))

    status = OptimizerStatus(state, cache(opt), value(problem(opt), x); config = config(opt))
    OptimizerResult(status, x, value(problem(opt), x))
end

function initialize_state!(state::OptimizerState)
    state 
end

const INITIAL_BFGS_X = 0.12345
const INITIAL_BFGS_G = 0.54321
const INITIAL_BFGS_F = 0.23456

function initialize_state!(state::Union{BFGSState{T}, DFPState{T}}) where {T}
    state.x̄ .= T(INITIAL_BFGS_X)
    state.ḡ .= T(INITIAL_BFGS_G)
    state.f̄ = T(INITIAL_BFGS_F)
    state.Q .= one(state.Q)

    state
end

function warn_iteration_number(state::OptimizerState, config::Options)
    if config.warn_iterations > 0 && iteration_number(state) ≥ config.warn_iterations
        println("WARNING: Optimizer took ", iteration_number(state), " iterations.")
    end
end