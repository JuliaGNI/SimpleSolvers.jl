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

    Newton(refactorize::Integer = 1) = new(refactorize)
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
function QuasiNewton(refactorize::Integer = DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER)
    Newton(refactorize)
end

"""
    NewtonSolver

A `const` derived from [`NonlinearSolver`](@ref) as `NewtonSolver{T} = NonlinearSolver{T,Newton}`.

# Constructors

The `NewtonSolver` can be called with a [`NonlinearProblem`](@ref) or with a `Callable`.

See [`NewtonSolver(::AbstractVector{T}, ::NonlinearProblem, ::AbstractVector{T}) where {T}`](@ref)
and [`NewtonSolver(::AbstractVector{T}, ::Callable, ::AbstractVector{T}) where {T}`](@ref).

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
- `jacobian_prototype`: the matrix whose storage and, if sparse, whose sparsity pattern are
  adopted by the Jacobian, the [`LinearProblem`](@ref) and the [`LinearSolver`](@ref) cache. It
  is *copied*, so the caller's matrix is left alone. A `SparseMatrixCSC` here is what runs a
  sparse Jacobian through the solver, and it requires `DF!` (see
  [`checkjacobianprototype`](@ref)) and a pattern that includes the diagonal if
  `regularization_factor` is non-zero. It also selects the default
  `linear_solver_method`; see [`default_linear_solver_method`](@ref),
- `refactorize::Int`: determines after how many steps the Jacobian is re-evaluated and refactored (see [`factorize!`](@ref)). `refactorize > 1` gives a quasi-Newton method (see [`QuasiNewton`](@ref)),
- `options_kwargs`: see [`Options`](@ref)
"""
const NewtonSolver{T} = NonlinearSolver{T, Newton}

# Behind a barrier because the constructor below is specialized on the closure types of the
# `NonlinearProblem` — see `report_linesearch_status`.
@noinline function report_static_refactorize(refactorize::Integer, config::Options)
    verbosity(config) ≥ 1 &&
        @warn "Static line search will not work with refactorize = $(refactorize). Setting refactorize = 1."
    nothing
end

function NewtonSolver(x::AT,
        nlp::NLST,
        ls::LST,
        linearsolver::LSoT,
        linesearch::LiSeT,
        cache::CT,
        config::Options{T};
        jacobian::Jacobian = JacobianAutodiff(nlp.F, x),
        refactorize::Integer = 1) where {
        T, AT <: AbstractVector{T}, NLST, LST, LSoT, LiSeT <: Linesearch{T}, CT}
    if refactorize > 1 && typeof(method(linesearch)) <: Static
        report_static_refactorize(refactorize, config)
        refactorize = 1
    end

    NonlinearSolver(x, nlp, ls, linearsolver, linesearch, cache, config;
        method = Newton(refactorize), jacobian = jacobian)
end

# Backwards-compatible form that builds the `Options` from keywords.  The `linesearch` handed
# in here carries an `Options` of its own (`Linesearch(problem, method)` defaults it); left as it
# comes, neither the solver's `verbosity` nor its `linesearch_max_iterations` would ever reach the
# line search.  It is therefore rebuilt on the solver's `Options` (see `with_config`): its problem
# and method are kept, only the options are replaced.
function NewtonSolver(x::AT,
        nlp::NLST,
        ls::LST,
        linearsolver::LSoT,
        linesearch::LiSeT,
        cache::CT;
        jacobian::Jacobian = JacobianAutodiff(nlp.F, x),
        refactorize::Integer = 1,
        options_kwargs...) where {
        T, AT <: AbstractVector{T}, NLST, LST, LSoT, LiSeT <: Linesearch{T}, CT}
    config = Options(T; options_kwargs...)
    NewtonSolver(x, nlp, ls, linearsolver, with_config(linesearch, config),
        cache, config; jacobian = jacobian, refactorize = refactorize)
end

"""
    NewtonSolver(x, nlp::NonlinearProblem, y = zero(x))

Build a [`NewtonSolver`](@ref) for the [`NonlinearProblem`](@ref) `nlp` with the initial
guess `x`, assembling the [`Jacobian`](@ref), the [`LinearProblem`](@ref), the
[`LinearSolver`](@ref), the [`Linesearch`](@ref) and the [`NonlinearSolverCache`](@ref).

`y` is a *prototype* for the residual ``F(x)``: it supplies a size and an element type, and
nothing that is computed from it survives (`alloc_j` turns it into a `NaN` matrix and the
cache stores `zero(y)`). It is not, however, left alone — [`JacobianAutodiff`](@ref) keeps it
as the buffer ForwardDiff writes the residual into, so a caller-supplied `y` is overwritten on
every Jacobian evaluation. It defaults to `zero(x)`, which is what a square system needs — and
every system here is square, since the [`LinearSolver`](@ref) factorizes the Jacobian.

!!! info
    The default is `zero(x)` rather than `similar(x)` because `alloc_j` broadcasts over `y`:
    for an element type whose `similar` leaves undefined references (`BigFloat`, say) an
    uninitialized prototype throws an `UndefRefError`. It also assumes `zero(x)` has the same
    type as `x`, which [`NonlinearSolverCache`](@ref) requires; for an `x` where it does not —
    a `SubArray`, whose `zero` is an `Array` — pass `y` explicitly.

The Jacobian stored in `nlp` (if any) takes precedence over autodiff, exactly as the `DF!`
keyword of [`NewtonSolver(::AbstractVector{T}, ::Callable, ::AbstractVector{T}) where {T}`](@ref)
does — see [`resolve_jacobian`](@ref).

# Keywords
- `linear_solver_method`
- `linesearch`
- `jacobian`
- `jacobian_prototype`
- `refactorize`
- `options_kwargs`: see [`Options`](@ref)

# Examples

```jldoctest; setup = :(using SimpleSolvers)
F(y, x, params) = y .= sin.(x) .^ 2
x = ones(3)
nlp = NonlinearProblem(F, x)

NewtonSolver(x, nlp) isa NewtonSolver

# output

true
```
"""
function NewtonSolver(
        x::AbstractVector{T}, nlp::NonlinearProblem, y::AbstractVector{T} = zero(x);
        linear_solver_method = missing, linesearch = Backtracking(T), jacobian = missing,
        jacobian_prototype = alloc_j(x, y), refactorize = 1, options_kwargs...) where {T}
    # The `Options` are built here, once, and shared by the solver *and* its line search, so
    # that `NewtonSolver(…; verbosity = 0)` silences the line search too and the inner ladder
    # is bounded by `linesearch_max_iterations` from the same place.
    config = Options(T; options_kwargs...)
    jacobian = resolve_jacobian(nlp.F, nlp.J, jacobian, x, y)
    checkjacobianprototype(jacobian, jacobian_prototype)
    cache = NonlinearSolverCache(x, y, jacobian_prototype)
    linearproblem = LinearProblem(jacobian_prototype)
    # From the prototype, not from `y`: `LinearSolver(method, ::AbstractVector)` allocates a
    # dense `zeros(T, n, n)`, which threw away the storage the Jacobian actually has and made
    # a sparse solve impossible. It also means a non-square Jacobian is now refused by
    # `checksquare` instead of silently sizing the solver from `length(y)`.
    linearsolver = LinearSolver(
        resolve_linear_solver_method(linear_solver_method,
            matrix(linearproblem)),
        matrix(linearproblem))
    ls = Linesearch(linesearch_problem(nlp, jacobian, cache), linesearch, config)
    NewtonSolver(x, nlp, linearproblem, linearsolver, ls, cache, config;
        jacobian = jacobian, refactorize = refactorize)
end

"""
    NewtonSolver(x, F, y)

# Keywords
- `linear_solver_method`
- `DF!`
- `linesearch`
- `jacobian`
- `jacobian_prototype`
- `refactorize`
- `options_kwargs`: see [`Options`](@ref)

The `Callable` `F` (and the optional `DF!`) are wrapped in a [`NonlinearProblem`](@ref), so
this is [`NewtonSolver(::AbstractVector{T}, ::NonlinearProblem, ::AbstractVector{T}) where {T}`](@ref)
with the problem built for the caller.
"""
function NewtonSolver(x::AbstractVector{T}, F::Callable, y::AbstractVector{T};
        (DF!) = missing, kwargs...) where {T}
    NewtonSolver(x, NonlinearProblem(F, DF!, x, y), y; kwargs...)
end

function NewtonSolver(x::AT, y::AT; F = missing, kwargs...) where {
        T, AT <: AbstractVector{T}}
    !ismissing(F) || error("You have to provide an F.")
    NewtonSolver(x, F, y; kwargs...)
end

"""
    direction!(d, x, s, params, iteration; stalled=false)

Compute the Newton direction (for the [`NewtonSolver`](@ref)). `stalled` is forwarded to
[`maybe_refactorize!`](@ref); see [`needs_refresh`](@ref).
"""
function direction!(d::AbstractVector{T}, x::AbstractVector{T}, s::NewtonSolver{T},
        params, iteration; stalled::Bool = false) where {T}
    # first we update the rhs of the linearproblem
    value!(rhs(linearproblem(s)), nonlinearproblem(s), x, params)
    rhs(linearproblem(s)) .*= -1
    # for a quasi-Newton method the Jacobian isn't updated in every iteration
    # (see `maybe_refactorize!`), unless the previous step stalled.
    maybe_refactorize!(s, x, params, iteration; stalled = stalled)
    ldiv!(d, linearsolver(s), rhs(linearproblem(s)))
end

function direction!(
        s::NewtonSolver, x::AbstractVector, params, iteration; stalled::Bool = false)
    direction!(direction(cache(s)), x, s, params, iteration; stalled = stalled)
end

# check_jacobian / print_jacobian operate on the Jacobian matrix, not the Jacobian
# functor.  An optional leading `io` argument is forwarded through (default stdout).
function check_jacobian(io::IO, s::NewtonSolver; kwargs...)
    check_jacobian(io, jacobianmatrix(s); kwargs...)
end
check_jacobian(s::NewtonSolver; kwargs...) = check_jacobian(jacobianmatrix(s); kwargs...)
print_jacobian(io::IO, s::NewtonSolver) = print_jacobian(io, jacobianmatrix(s))
print_jacobian(s::NewtonSolver) = print_jacobian(jacobianmatrix(s))

function NonlinearSolver(method::Newton, args...; kwargs...)
    NewtonSolver(args...; refactorize = method.refactorize, kwargs...)
end
