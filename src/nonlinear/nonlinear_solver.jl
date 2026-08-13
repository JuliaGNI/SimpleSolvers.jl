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
    NonlinearSolver

A `struct` that comprises *Newton solvers* (see [`Newton`](@ref)), the *Picard solver* (also known as fixed-point iteration; see [`Picard`](@ref)) and the *Dogleg solver* (see [`DogLeg`](@ref)).

!!! info
    The associated solvers are `const`s derived from `NonlinearSolver`. See [`NewtonSolver`](@ref), [`PicardSolver`](@ref) and [`DogLegSolver`](@ref). In practice we usually call those associated constructors directly rather than creating a `NonlinearSolver` instance manually.

# Keys

- `nonlinearproblem::`[`NonlinearProblem`](@ref): the system that has to be solved. This can be accessed by calling [`nonlinearproblem`](@ref),
- `linearproblem::`[`LinearProblem`](@ref),
- `jacobian::`[`Jacobian`](@ref): the Jacobian is used to compute the *direction* in the solver step (see [`solver_step!`](@ref)). This can be accessed by calling [`jacobian`](@ref),
- `linearsolver::`[`LinearSolver`](@ref): the linear solver is used to compute the *direction* of the solver step (see [`solver_step!`](@ref)). This can be accessed by calling [`linearsolver`](@ref),
- `linesearch::`[`Linesearch`](@ref)
- `method::`[`NonlinearSolverMethod`](@ref): the solver method (e.g. [`Newton`](@ref)),
- `cache::`[`NonlinearSolverCache`](@ref)
- `config::`[`Options`](@ref)
"""
struct NonlinearSolver{T,MT<:NonlinearSolverMethod,NLST<:NonlinearProblem,LST<:AbstractLinearProblem,JT<:Jacobian{T},LSoT<:AbstractLinearSolver,LiSeT<:Linesearch{T},CT<:AbstractNonlinearSolverCache{T}} <: AbstractSolver
    nonlinearproblem::NLST
    linearproblem::LST
    jacobian::JT
    linearsolver::LSoT
    linesearch::LiSeT
    method::MT

    cache::CT
    config::Options{T}

    # No `options_kwargs...` here: the `Options` arrive ready-made as `config`, so a keyword
    # sink would only swallow misspelled option names silently.
    function NonlinearSolver(x::AT, nlp::NLST, ls::LST, linearsolver::LSoT, linesearch::LiSeT, cache::CT, config::Options{T}; method::MT=Newton(), jacobian::JT=JacobianAutodiff(nlp.F, x)) where {T,AT<:AbstractVector{T},MT<:NonlinearSolverMethod,JT<:Jacobian{T},NLST<:NonlinearProblem,LST<:AbstractLinearProblem,LSoT<:AbstractLinearSolver,LiSeT<:Linesearch{T},CT<:AbstractNonlinearSolverCache{T}}
        new{T,MT,NLST,LST,JT,LSoT,LiSeT,CT}(nlp, ls, jacobian, linearsolver, linesearch, method, cache, config)
    end
end

cache(s::NonlinearSolver) = s.cache
config(s::NonlinearSolver) = s.config
method(s::NonlinearSolver) = s.method

linearproblem(s::NonlinearSolver) = s.linearproblem
linesearch(s::NonlinearSolver) = s.linesearch
jacobian(s::NonlinearSolver) = s.jacobian

function initialize!(s::NonlinearSolver, x::AbstractVector)
    initialize!(cache(s), x)

    s
end

"""
    nonlinearproblem(solver)

Return the [`NonlinearProblem`](@ref) contained in the [`NonlinearSolver`](@ref). Compare this to [`linearsolver`](@ref).
"""
nonlinearproblem(s::NonlinearSolver) = s.nonlinearproblem

jacobian!(s::NonlinearSolver{T}, x::AbstractVector{T}, params) where {T} = jacobian(s)(jacobianmatrix(cache(s)), x, params)

"""
    jacobianmatrix(solver::NonlinearSolver)

Return the evaluated Jacobian (a matrix) stored in the cache (see [`NonlinearSolverCache`](@ref)) of `solver`.

Also see [`jacobian(::NonlinearProblem)`](@ref).
"""
jacobianmatrix(solver::NonlinearSolver) = jacobianmatrix(cache(solver))

"""
    linearsolver(solver)

Return the linear part (i.e. a [`LinearSolver`](@ref)) of an [`NewtonSolver`](@ref).

# Examples

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: linearsolver)
x = rand(3)
y = rand(3)
F(x) = tanh.(x)
F!(y, x, params) = y .= F(x)
s = NewtonSolver(x, y; F = F!)
linearsolver(s)

# output

LinearSolver{Float64, LU{Missing}, SimpleSolvers.LUSolverCache{Float64, StaticArraysCore.MMatrix{3, 3, Float64, 9}}}(LU{Missing}(missing, true), SimpleSolvers.LUSolverCache{Float64, StaticArraysCore.MMatrix{3, 3, Float64, 9}}([0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0], [0, 0, 0], [0, 0, 0], 0))
```
"""
linearsolver(solver::NonlinearSolver) = solver.linearsolver


struct NonlinearSolverException <: Exception
    msg::String
end

Base.showerror(io::IO, e::NonlinearSolverException) = print(io, "Nonlinear Solver Exception: ", e.msg, "!")

"""
    resolve_jacobian(F, DF!, jacobian, x, y)

Resolve the [`Jacobian`](@ref) for a nonlinear-solver constructor: an explicit `DF!`
wins (wrapped as a [`JacobianFunction`](@ref)), otherwise an explicit `jacobian`,
otherwise a lazily-built [`JacobianAutodiff`](@ref). Building the autodiff Jacobian
lazily avoids allocating a ForwardDiff config when either `DF!` or a `jacobian` is
supplied.
"""
function resolve_jacobian(F, DF!, jacobian, x::AbstractVector{T}, y) where {T}
    ismissing(DF!) || return JacobianFunction{T}(F, DF!)
    ismissing(jacobian) ? JacobianAutodiff(F, x, y) : jacobian
end

"""
    maybe_refactorize!(s, x, params, iteration; force=false, stalled=false)

Re-evaluate the [`Jacobian`](@ref) at `x`, copy it into the [`LinearProblem`](@ref)
(adding the diagonal `regularization_factor`), and refactorize the
[`LinearSolver`](@ref) — but only on a refactorization step: a fresh state or the
first step (`iteration ≤ 1`), every `refactorize` iterations (see [`Newton`](@ref)),
when the previous step made no progress (`stalled`, see [`needs_refresh`](@ref)), or
when `force`d (used by the [`DogLegSolver`](@ref) to recover from a collapsed
trust-region radius). Otherwise the stale Jacobian and its factorization are reused
(quasi-Newton). Returns the solver `s`.

`stalled` is what makes the quasi-Newton mode safe to combine with `max_stalls`: a step
that did not move the iterate would otherwise rebuild the *same* direction from the same
stale Jacobian on the next `refactorize - 1` iterations and reproduce the same negligible
step, so the solve would be given up on (see [`stalled_step`](@ref)) for a reason a fresh
Jacobian could have fixed. Refreshing immediately means the second consecutive stall is
one that a fresh Jacobian did *not* fix, which is the conclusive evidence `max_stalls = 2`
assumes. It is also the response [`check_anchor`](@ref) prescribes for an ascent anchor,
which is a stale-Jacobian symptom.
"""
function maybe_refactorize!(s::NonlinearSolver, x, params, iteration; force::Bool=false, stalled::Bool=false)
    (force || stalled || mod(iteration, method(s).refactorize) == 0 || iteration ≤ 1) || return s
    jacobian!(s, x, params)
    lp = linearproblem(s)
    matrix(lp) .= jacobianmatrix(s)
    idxs = diagind(matrix(lp))
    @view(matrix(lp)[idxs]) .+= config(s).regularization_factor
    factorize!(linearsolver(s), lp)
    s
end

# Behind a barrier because `nan_recovery!` below is specialized on the `NonlinearSolver`, hence on
# its problem's closure types — see `report_linesearch_status`.
@noinline function report_nan_direction(config::Options)
    verbosity(config) ≥ 2 && @warn "Non-finite value (NaN or Inf) at the trial iterate. Reducing length of direction vector."
    nothing
end

"""
    nan_recovery!(s, x, params)

Damp `direction(cache(s))` by `nan_factor` until the trial iterate `x + d` has a
finite residual (or the `nan_max_iterations` budget of dampings is exhausted). On return
`solution(cache(s))` and `value(cache(s))` hold the last trial iterate and its residual, and
`direction(cache(s))` is the direction that trial was built from — so `value(cache(s))` is
`F(x + direction(cache(s)))` on every exit path, which is what the Picard
[`solver_step!`](@ref) reads instead of re-evaluating `F`. One trial is always evaluated, so a
budget of zero refreshes the cache without damping at all. Used by the generic and Picard
[`solver_step!`](@ref)s. Returns the solver `s`.

"Finite" means `isfinite`, not merely "not `NaN`": a residual that has *overflowed* is
as unusable as an undefined one and is just as much a symptom of a step that left the
region where `F` is representable — the `-1/|x|` of issue #130 gives `Inf` at `x = 0`,
not `NaN`. The option names (`nan_factor`, `nan_max_iterations`) predate that and are
kept for compatibility.

Damping is only meaningful because the *direction* is known to be finite: the callers
reject a non-finite one outright (see [`solver_step!`](@ref)), and they must, since
`Inf * nan_factor` is `Inf` and the loop would spend its whole budget reproducing the
same trial iterate.
"""
function nan_recovery!(s::NonlinearSolver{T}, x, params) where {T}
    # The damping happens *before* the trial it is for, not after the one that failed. That is what
    # makes both post-conditions hold on every exit path: the budget bounds the *damping* (a budget
    # of zero damps not at all), and `value(cache(s))` is the residual of the trial built from the
    # direction the cache currently holds — damping after the last trial would leave the two one
    # factor apart, and the Picard step reads them as a pair. The `i = 0` pass evaluates one trial
    # unconditionally: a budget of zero used to skip the body entirely and leave the cache holding
    # whatever the previous solver step had put there, which the Picard step then read as if it were
    # this step's trial, committing a stale iterate.
    for i in 0:config(s).nan_max_iterations
        if i > 0
            report_nan_direction(config(s))
            direction(cache(s)) .*= T(config(s).nan_factor)
        end
        solution(cache(s)) .= x .+ direction(cache(s))
        value!(value(cache(s)), nonlinearproblem(s), solution(cache(s)), params)
        all(isfinite, value(cache(s))) && break
    end
    s
end

"""
    solver_step!(x, s, state, params)

Compute one step for solving the problem stored in an instance `s` of [`NonlinearSolver`](@ref).

# Examples

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: solver_step!, NullParameters)
julia> f(y, x, params) = y .= sin.(x .- .5) .^ 2
f (generic function with 1 method)

julia> x = ones(3) / 4
3-element Vector{Float64}:
 0.25
 0.25
 0.25

julia> y = zero(x)
3-element Vector{Float64}:
 0.0
 0.0
 0.0

julia> s = NewtonSolver(x, similar(x); F = f);

julia> state = NonlinearSolverState(x); update!(state, x, f(y, x, NullParameters()));

julia> solver_step!(x, s, state, NullParameters())
3-element Vector{Float64}:
 0.37767096061051814
 0.37767096061051814
 0.37767096061051814
```
"""
function solver_step!(x::AbstractVector{T}, s::NonlinearSolver{T}, state::NonlinearSolverState{T}, params) where {T}
    # A previous step that made no progress gets a freshly evaluated Jacobian rather than the
    # stale one that produced it (see `maybe_refactorize!`).
    direction!(s, x, params, iteration_number(state); stalled=needs_refresh(state))
    # A non-finite direction is not a step that can be shortened — `nan_recovery!` damps by a
    # factor and `Inf * nan_factor` is still `Inf` — so it is rejected outright rather than
    # handed to a recovery that cannot work. This is also what makes the recovery below sound:
    # past this line the direction is finite and each damping actually shortens the trial step.
    all(isfinite, direction(cache(s))) || throw(NonlinearSolverException("non-finite direction vector"))

    nan_recovery!(s, x, params)

    # `params.φ₀` hands the line search the merit at the α = 0 anchor, which the solver has
    # just evaluated at the current iterate — see `linesearch_problem`.
    lsparams = (x=x, parameters=params, φ₀=L2norm(value(state)))
    lsstatus = solve_with_status(linesearch(s), one(T), lsparams)
    linesearch_warnings(lsstatus, linesearch(s), lsparams)

    # A line search that reports the merit's round-off floor or a non-descent anchor knows the
    # iteration cannot make progress along this direction, one iteration before the step-based
    # diagnosis of `stalled_step` sees it. Recording it here forces a fresh Jacobian on the next
    # step and gives up after `max_stalls` if that does not help.
    nodescent = outcome(lsstatus) === LINESEARCH_NO_DESCENT
    (isfloor(lsstatus) || nodescent) && flag_stall!(state)

    # The step is not taken along a direction the line search has *rejected outright*: no α can
    # decrease the merit along an ascent direction, so moving would only make the forced
    # refactorization start from a worse point. (The `LINESEARCH_FLOOR` step *is* taken — it is
    # the smallest informative one, and it is not an ascent direction.) The line search itself
    # still returns α > 0 as its contract requires; whether to use it is the caller's call.
    nodescent && return x

    compute_new_iterate!(x, steplength(lsstatus), direction(cache(s)))

    x
end

"""
    solve!(x, s, state)

Solve the [`NonlinearProblem`](@ref) contained in the [`NonlinearSolver`](@ref) with the initial condition `x`.

You also have to supply a [`NonlinearSolverState`](@ref).
"""
function solve!(x::AbstractArray, s::NonlinearSolver, state::NonlinearSolverState, params=NullParameters())
    initialize!(s, x)
    initialize!(state, x, value!(value(cache(s)), nonlinearproblem(s), x, params))

    # The stopping criteria are tested *before* the first step as well. An initial guess that
    # already satisfies them must not be perturbed by a full solver step — including a line
    # search asked to improve an already-exact residual, which is one source of the spurious
    # "did not satisfy the sufficient decrease condition" warnings. Only the absolute branch is
    # reachable at iteration 0: `rxₛ` and `rfₛ` are `NaN` until the first `update!` (see
    # `initialize!`), so this fires exactly when ‖F(x₀)‖ ≤ f_abstol — for the default
    # `f_abstol = 0` only at an exact root. A caller who insists on at least one iteration
    # (`min_iterations ≥ 1`) still gets one, and a `NaN` initial residual still gets a step
    # (the `havenonfinite` branch requires `iterations ≥ 1`).
    while !meets_stopping_criteria(state, config(s))
        increase_iteration_number!(state)
        solver_step!(x, s, state, params)
        update!(state, x, value!(value(cache(s)), nonlinearproblem(s), x, params))
        record_iteration!(state, config(s))
    end

    status = NonlinearSolverStatus(state, config(s))
    nonlinear_solver_warnings(status, config(s))
    config(s).verbosity > 1 && print_status(status, config(s))

    x
end

"""
    status(solver, state)

Return the [`NonlinearSolverStatus`](@ref) for the [`NonlinearSolverState`](@ref) `state` as
assessed with the [`Options`](@ref) of `solver`.

[`solve!`](@ref) returns the solution `x` (updated in place), not a status, so this is how a
caller inspects the *outcome* of a solve — in particular whether it converged
([`isconverged`](@ref)) or merely stagnated at the residual floor ([`isstalled`](@ref)). The
state is the caller's own object (it is passed to `solve!`), so nothing has to be threaded back
out of the solve.

# Examples

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: isconverged, status)
julia> F(y, x, params) = y .= x .^ 2 .- 2;

julia> x = [1.0]; s = NewtonSolver(x, similar(x); F = F, verbosity = 0);

julia> state = SolverState(s);

julia> solve!(x, s, state);

julia> isconverged(status(s, state))
true
```
"""
status(s::NonlinearSolver, state::NonlinearSolverState) = NonlinearSolverStatus(state, config(s))

solve!(x::AbstractArray, s::NonlinearSolver, params=NullParameters()) = solve!(x, s, NonlinearSolverState(x, value(cache(s))), params)

"""
    solve!(x, prob, method, args...; kwargs...)

Solve the [`NonlinearProblem`](@ref) `prob` with the [`NonlinearSolverMethod`](@ref) `method`,
starting from `x` and overwriting it with the solution (which is returned).

The remaining positional arguments are passed on to
[`solve!(::AbstractArray, ::NonlinearSolver, ::NonlinearSolverState, params)`](@ref), so they are
either nothing at all, the `params` of the problem, or a [`NonlinearSolverState`](@ref) followed by
`params`.

The keyword arguments are the [`Options`](@ref) keywords together with whatever the constructor
selected by `method` accepts — which is *not* the same set for all four methods:

| `method` | constructor keywords |
|:---|:---|
| [`Newton`](@ref), [`QuasiNewton`](@ref) | `linesearch`, `jacobian`, `linear_solver_method` |
| [`DogLeg`](@ref) | `jacobian`, `linear_solver_method` |
| [`Picard`](@ref) | `jacobian` |

`Picard` and `DogLeg` consult no line search, so they reject a `linesearch` keyword rather than
ignoring it (it falls through to [`Options`](@ref) and raises a `MethodError` there) — see
[`PicardSolver(::AbstractVector{T}, ::NonlinearProblem, ::AbstractVector{T}) where {T}`](@ref).

`refactorize` is deliberately absent from that table: it belongs to the `method`, as `Newton(5)`
(equivalently [`QuasiNewton`](@ref)) or `DogLeg(5)`. Passing it as a keyword does work for those
two — it is forwarded after `method.refactorize` and, being the rightmost occurrence, silently
wins — but `Picard` has no Jacobian to refactorize and errors on it. Configure it through `method`.

!!! info
    This is the convenience path: every call *builds a solver* — a Jacobian (with its ForwardDiff
    configuration), a factorization cache and the line-search buffers. Code that solves repeatedly
    should construct one [`NonlinearSolver`](@ref) and call
    [`solve!(::AbstractArray, ::NonlinearSolver, ::NonlinearSolverState, params)`](@ref) on it
    instead.

`solve!` returns the solution, not a status; use [`solve_with_status!`](@ref) if the outcome of the
solve is needed.

# Examples

```jldoctest; setup = :(using SimpleSolvers)
julia> F(y, x, params) = y .= x .^ 2 .- 2;

julia> prob = NonlinearProblem(F, zeros(1));

julia> x = [1.0];

julia> solve!(x, prob, Newton(); verbosity = 0)
1-element Vector{Float64}:
 1.4142135623730951
```
"""
function solve!(x::AbstractVector, prob::NonlinearProblem, method::NonlinearSolverMethod, args...; kwargs...)
    solve!(x, NonlinearSolver(method, x, prob; kwargs...), args...)
end

"""
    solve(x, prob, method, args...; kwargs...)

Solve the [`NonlinearProblem`](@ref) `prob` with the [`NonlinearSolverMethod`](@ref) `method`,
starting from the initial guess `x`, and return the solution as a *new* array — `x` itself is left
untouched. This is [`solve!(::AbstractVector, ::NonlinearProblem, ::NonlinearSolverMethod)`](@ref)
on a copy, and takes the same arguments; the note on solver construction there applies here too.

The initial guess has to be passed because a `NonlinearProblem` stores neither the solution nor the
residual, so nothing else determines the size and type of the array to allocate.

# Examples

```jldoctest; setup = :(using SimpleSolvers)
julia> F(y, x, params) = y .= x .^ 2 .- 2;

julia> prob = NonlinearProblem(F, zeros(1));

julia> x₀ = [1.0];

julia> solve(x₀, prob, Newton(); verbosity = 0)
1-element Vector{Float64}:
 1.4142135623730951

julia> x₀
1-element Vector{Float64}:
 1.0
```
"""
function solve(x::AbstractVector, prob::NonlinearProblem, method::NonlinearSolverMethod, args...; kwargs...)
    solve!(copy(x), prob, method, args...; kwargs...)
end

"""
    solve_with_status!(x, s, params=NullParameters())
    solve_with_status!(x, prob, method, params=NullParameters(); kwargs...)

Solve as [`solve!`](@ref) does — `x` is overwritten with the solution — but return the
[`NonlinearSolverStatus`](@ref) instead of `x`.

This is how the outcome of a solve is obtained without holding on to a
[`NonlinearSolverState`](@ref): the state is built here and handed to [`status`](@ref) afterwards.
The `!` is part of the name because `x` *is* modified, unlike in the line-search
[`solve_with_status`](@ref), whose `α` is a number.

!!! info
    The predicates that read the returned status — [`isconverged`](@ref) and
    [`isstalled`](@ref) — are **not** exported, for the same reason the line-search predicates
    are not: they are generic names a package doing `using SimpleSolvers` may well want for
    itself. Reach them as `SimpleSolvers.isconverged(st)`, or import them explicitly with
    `using SimpleSolvers: isconverged`.

# Examples

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: isconverged)
julia> F(y, x, params) = y .= x .^ 2 .- 2;

julia> prob = NonlinearProblem(F, zeros(1));

julia> x = [1.0];

julia> isconverged(solve_with_status!(x, prob, Newton(); verbosity = 0))
true
```
"""
function solve_with_status!(x::AbstractArray, s::NonlinearSolver, params=NullParameters())
    state = NonlinearSolverState(x, value(cache(s)))
    solve!(x, s, state, params)
    status(s, state)
end

function solve_with_status!(x::AbstractVector, prob::NonlinearProblem, method::NonlinearSolverMethod, params=NullParameters(); kwargs...)
    solve_with_status!(x, NonlinearSolver(method, x, prob; kwargs...), params)
end
