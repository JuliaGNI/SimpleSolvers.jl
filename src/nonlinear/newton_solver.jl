"""
    Newton(refactorize=1)

The Newton (and quasi-Newton) nonlinear solver method.

# Constructors

```jldoctest; setup = :(using SimpleSolvers)
Newton()

# output

Newton(1)
```

```jldoctest; setup = :(using SimpleSolvers)
QuasiNewton()

# output

Newton(5)
```
!!! info
    The *refactorize* parameter determines how often the [`Jacobian`](@ref) is
    re-evaluated and refactored (see [`factorize!`](@ref)). The default
    `refactorize = 1` refactorizes on every step (a plain Newton method), whereas
    `refactorize > 1` reuses the factorization in between, giving a quasi-Newton
    method (conveniently constructed via [`QuasiNewton`](@ref)).
"""
struct Newton <: NonlinearSolverMethod
    refactorize::Int

    Newton(refactorize::Integer=1) = new(refactorize)
end

"""
The default number of iterations before the [`Jacobian`](@ref) is refactored when
constructing a quasi-Newton method via [`QuasiNewton`](@ref).
"""
const DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER = 5

"""
    QuasiNewton(refactorize=$(DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER))

Convenience constructor for a [`Newton`](@ref) method whose [`Jacobian`](@ref) is
only re-evaluated and refactored every `refactorize` iterations. Equivalent to
`Newton(refactorize)` but with a quasi-Newton default (see
[`DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER`](@ref)).
"""
QuasiNewton(refactorize::Integer=DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER) = Newton(refactorize)

"""
    NewtonSolver

A `const` derived from [`NonlinearSolver`](@ref) as `NewtonSolver{T} = NonlinearSolver{T,Newton}`.

# Constructors

The `NewtonSolver` can be called with a [`NonlinearProblem`](@ref) or with a `Callable`.

See [`NewtonSolver(::AbstractVector{T}, ::Callable, ::AbstractVector{T}) where {T}`](@ref).

```jldoctest; setup = :(using SimpleSolvers)
F(y, x, params) = y .= sin.(x) ^ 2
x = ones(5)
y = zeros(5)

ns = NewtonSolver(x, F, y)
typeof(ns) <: NewtonSolver

# output

true
```

# Keywords
- `linear_solver_method`: the method used to build the linear solver (see [`LinearSolver`](@ref)) that computes the *direction* of the solver step (see [`solver_step!`](@ref)),
- `DF!`: an in-place function computing the Jacobian,
- `linesearch::`[`Linesearch`](@ref)
- `jacobian::`[`Jacobian`](@ref)
- `refactorize::Int`: determines after how many steps the Jacobian is re-evaluated and refactored (see [`factorize!`](@ref)). `refactorize > 1` gives a quasi-Newton method (see [`QuasiNewton`](@ref)),
- `options_kwargs`: see [`Options`](@ref)
"""
const NewtonSolver{T} = NonlinearSolver{T,Newton}

function NewtonSolver(x::AT, nlp::NLST, ls::LST, linearsolver::LSoT, linesearch::LiSeT, cache::CT; jacobian::Jacobian=JacobianAutodiff(nlp.F, x), refactorize::Integer=1, options_kwargs...) where {T,AT<:AbstractVector{T},NLST,LST,LSoT,LiSeT,CT}
    config = Options(T; options_kwargs...)

    if refactorize > 1 && typeof(method(linesearch)) <: Static
        config.verbosity ≥ 1 && (@warn "Static line search will not work with refactorize = $(refactorize). Setting refactorize = 1.")
        refactorize = 1
    end

    NonlinearSolver(x, nlp, ls, linearsolver, linesearch, cache, config; method=Newton(refactorize), jacobian=jacobian)
end

"""
    NewtonSolver(x, F, y)

# Keywords
- `linear_solver_method`
- `DF!`
- `linesearch`
- `jacobian`
- `refactorize`
- `options_kwargs`: see [`Options`](@ref)
"""
function NewtonSolver(x::AbstractVector{T}, F::Callable, y::AbstractVector{T}; linear_solver_method=LU(), (DF!)=missing, linesearch=Backtracking(T), jacobian=missing, refactorize=1, kwargs...) where {T}
    nlp = NonlinearProblem(F, DF!, x, y)
    # Build the default autodiff Jacobian lazily, so we don't allocate ForwardDiff
    # configs when either a Jacobian or a `DF!` is supplied.
    jacobian = ismissing(DF!) ?
               (ismissing(jacobian) ? JacobianAutodiff(F, x, y) : jacobian) :
               JacobianFunction{T}(F, DF!)
    cache = NonlinearSolverCache(x, y)
    linearproblem = LinearProblem(alloc_j(x, y))
    linearsolver = LinearSolver(linear_solver_method, y)
    ls = Linesearch(linesearch_problem(nlp, jacobian, cache), linesearch)
    NewtonSolver(x, nlp, linearproblem, linearsolver, ls, cache; jacobian=jacobian, refactorize=refactorize, kwargs...)
end

function NewtonSolver(x::AT, y::AT; F=missing, kwargs...) where {T,AT<:AbstractVector{T}}
    !ismissing(F) || error("You have to provide an F.")
    NewtonSolver(x, F, y; kwargs...)
end

"""
    direction!(d, x, s, params, iteration)

Compute the Newton direction (for the [`NewtonSolver`](@ref)).
"""
function direction!(d::AbstractVector{T}, x::AbstractVector{T}, s::NewtonSolver{T}, params, iteration) where {T}
    # first we update the rhs of the linearproblem
    value!(rhs(linearproblem(s)), nonlinearproblem(s), x, params)
    rhs(linearproblem(s)) .*= -1
    # for a quasi-Newton method the Jacobian isn't updated in every iteration:
    # factorize on a fresh state (iteration 0) and the first step (iteration 1),
    # then every `refactorize` iterations.
    if (mod(iteration, method(s).refactorize) == 0 || iteration ≤ 1)
        jacobian!(s, x, params)
        matrix(linearproblem(s)) .= jacobianmatrix(s)
        idxs = diagind(matrix(linearproblem(s)))
        @view(matrix(linearproblem(s))[idxs]) .+= config(s).regularization_factor
        factorize!(linearsolver(s), linearproblem(s))
    end
    ldiv!(d, linearsolver(s), rhs(linearproblem(s)))
end

function direction!(s::NewtonSolver, x::AbstractVector, params, iteration)
    direction!(direction(cache(s)), x, s, params, iteration)
end

# check_jacobian / print_jacobian operate on the Jacobian matrix, not the Jacobian
# functor.  An optional leading `io` argument is forwarded through (default stdout).
check_jacobian(io::IO, s::NewtonSolver; kwargs...) = check_jacobian(io, jacobianmatrix(s); kwargs...)
check_jacobian(s::NewtonSolver; kwargs...) = check_jacobian(jacobianmatrix(s); kwargs...)
print_jacobian(io::IO, s::NewtonSolver) = print_jacobian(io, jacobianmatrix(s))
print_jacobian(s::NewtonSolver) = print_jacobian(jacobianmatrix(s))

NonlinearSolver(method::Newton, args...; kwargs...) = NewtonSolver(args...; refactorize=method.refactorize, kwargs...)
