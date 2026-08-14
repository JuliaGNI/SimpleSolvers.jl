"""
    LinesearchMethod{T} <: SolverMethod

Examples include [`Static`](@ref), [`Backtracking`](@ref), [`Bisection`](@ref) , [`BierlaireQuadratic`](@ref) and [`Quadratic`](@ref).
See these examples for specific information on linesearch algorithms.

# Extended help

A `LinesearchMethod` is usually used in [`Linesearch`](@ref) (or with [`solve`](@ref)).

It is a subtype of `SolverMethod` (imported from `GeometricBase`) — line searches
are one-dimensional subproblems used *inside* nonlinear solvers and
optimizers, so (unlike a [`NonlinearSolverMethod`](@ref)) a `LinesearchMethod` is
not itself a nonlinear-solver method.

# The line search contract

Every method reached through [`solve`](@ref) or [`solve_with_status`](@ref) guarantees:

1. **It never throws.** A situation it cannot handle is *reported*, never raised — a line
   search must not abort the enclosing solve. Bracketing helpers signal failure with `nothing`
   (see [`bracket_minimum`](@ref), [`triple_point_finder`](@ref)) and the method maps that onto
   a [`LinesearchOutcome`](@ref). The one deliberate exception is a caller error rather than a
   situation arising from the merit: a `params.αmax` that is not a usable ceiling raises an
   `ArgumentError` before any evaluation — see [`linesearch_αmax`](@ref).
2. **It returns ``\\alpha > 0``.** Never the ``\\alpha = 0`` anchor, which would freeze the
   outer iterate (`x .+= 0 .* d`), and never a negative step: ``\\alpha`` scales a direction
   that has already been chosen, so its sign is not the line search's to decide.
3. **It reports through [`linesearch_warnings`](@ref) only** — one message site and one
   verbosity policy for all methods (genuine failure at `verbosity ≥ 1`; the benign
   round-off-floor and stationary outcomes at `≥ 2`). And it reports there only when the *user*
   called it: a program calls [`solve_with_status`](@ref), acts on the
   [`LinesearchStatus`](@ref) and sees no messages at all. A [`NonlinearSolver`](@ref) is such a
   program — see [`record_linesearch!`](@ref). This is structural rather than a convention a
   method has to keep: a method implements `solve_with_status`, and [`solve`](@ref) is derived
   from it once, for all methods, as that call plus the report.
4. **A non-finite or ascending anchor is reported, not assumed away** — see
   [`check_anchor`](@ref).
5. **It terminates in a bounded number of merit evaluations, independently of the merit's
   scale.** Multiplying ``\\varphi`` by a constant must not change the cost.
6. **It returns ``\\alpha \\leq \\alpha_\\mathrm{max}``** — see [`linesearch_αmax`](@ref) for the
   two ways that ceiling is set and why one of them has to be per call.

# The two families

What is *not* standardised is the meaning of the input ``\\alpha`` and what each method
guarantees about the step, because there are two distinct kinds:

- **Condition-satisfying, ``\\alpha``-relative** — [`Backtracking`](@ref),
  [`StrongWolfe`](@ref) and trivially [`Static`](@ref): "given the trial step ``\\alpha``,
  return a step satisfying a decrease condition". The result depends on the input ``\\alpha``.
  [`Backtracking`](@ref) shrinks it only, and therefore returns it unchanged whenever it is
  acceptable, unless its `expand` key is set; [`StrongWolfe`](@ref) brackets in both directions.
- **Minimising, ``\\alpha``-independent** — [`Bisection`](@ref), [`Quadratic`](@ref) and
  [`BierlaireQuadratic`](@ref): "approximate the minimiser of ``\\varphi`` along the
  direction". The input ``\\alpha`` only seeds the bracketing (see issue #164), and no Wolfe
  condition is checked.
"""
abstract type LinesearchMethod{T} <: SolverMethod end

Base.eltype(::LinesearchMethod{T}) where {T} = T

@doc raw"""
    method_αmax(method)

The largest step length the [`LinesearchMethod`](@ref) `method` will try of its own accord, i.e.
its `αmax` field where it has one. The fallback is `Inf`, which is what [`Backtracking`](@ref)
returns: it *shrinks* the caller's trial step, so the trial step is already its ceiling, and its
opt-in expansion phase carries its own bound (`q^nexpand`).

[`Bisection`](@ref), [`Quadratic`](@ref), [`BierlaireQuadratic`](@ref) and [`StrongWolfe`](@ref)
each have the field, defaulting to [`DEFAULT_LINESEARCH_αmax`](@ref). Use [`linesearch_αmax`](@ref)
rather than this, which is only half of the ceiling.
"""
method_αmax(::LinesearchMethod{T}) where {T} = T(Inf)

@doc raw"""
    linesearch_αmax(method, params)

The ceiling on the step length a [`LinesearchMethod`](@ref) may return: the smaller of the
method's own [`method_αmax`](@ref) and the caller's `params.αmax`, if the caller supplied one.
This is the single definition of that policy, as [`check_anchor`](@ref) is of the anchor policy,
and every built-in method calls it once before it searches.

# Why the caller's half has to be per call

A bracketing search grows its bracket outward until the merit stops falling, so where ``\varphi``
is nearly flat — or its minimiser genuinely far away — the step it returns is bounded only by the
bracketing budget: ``2^{100}`` times the initial step, in principle. In a Euclidean problem that is
self-correcting, since ``\varphi`` grows with ``\alpha`` and the search's own decrease test throws
such a step out. On a *compact* manifold it is not: ``\varphi`` is bounded there and can be
genuinely lower at ``\alpha = 10^9`` than at ``\alpha = 0``, so nothing the search can measure
calls the step too large. **The merit is not a bound on the step**, and the bound that does exist
belongs to the caller — for a manifold solver, the ``2\pi`` of a rotation, divided by the norm of
the direction, which changes at every solver step.

Hence the two halves. `method_αmax` is a backstop that costs nothing and needs no knowledge of the
problem; `params.αmax` is how a caller with a scale of its own imposes it, and it is read from
`params` rather than taken as a keyword so that the extension point
[`solve_with_status`](@ref)`(ls, α, params)` keeps its signature — a third-party method that does
not know about the ceiling still compiles, and one that does needs only this call. It is the same
channel `params.φ₀` uses (see [`linesearch_problem`](@ref)), and the `hasproperty` guard is
resolved at compile time, so a caller that supplies nothing pays nothing.

# Example

```julia
solve_with_status(ls, one(T), (x = x, parameters = params, αmax = 2π / norm(direction)))
```

The other fields are the ones the merit closures of [`linesearch_problem`](@ref) read; `αmax` is
additional and independent of them.

!!! info "A ceiling is not a failure"
    A search that stops at the ceiling returns it, with the merit *measured* there and classified
    by the usual round-off allowance ``\tau``. It is a `LINESEARCH_DECREASED` if the merit really
    did fall; see [`DEFAULT_LINESEARCH_αmax`](@ref).

!!! warning "An invalid ceiling raises"
    A `params.αmax` that is not positive, or is `NaN`, is a caller error rather than a situation
    arising from the merit, and raises an `ArgumentError` before a single evaluation is spent — as
    the method constructors do for their own parameters. This is the one deliberate exception to
    the never-throws clause of [`LinesearchMethod`](@ref), which is about the problem and not about
    the call. `Inf` is *not* an error: it says the caller has no scale of its own, and leaves the
    method's ceiling in place.
"""
linesearch_αmax(m::LinesearchMethod{T}, params) where {T} = min(method_αmax(m), caller_αmax(T, params))

# The caller's half of `linesearch_αmax`. `hasproperty` on a `NamedTuple` (and on the
# `NullParameters` singleton, where it is `false`) is resolved from the type, so this whole call
# constant-folds away for a caller that supplies no ceiling — which is what keeps a solve
# allocation-free and its merit-evaluation counts unchanged.
function caller_αmax(::Type{T}, params) where {T}
    hasproperty(params, :αmax) || return T(Inf)
    αmax = T(params.αmax)
    # `Inf` is accepted and means "no ceiling of my own", which leaves the method's in place — it
    # cannot produce an unbounded step by itself, and a caller deriving its ceiling from a scale
    # (`2π / ‖δ‖`, say) should not have to special-case a vanishing direction. `NaN` and a
    # non-positive value are neither: the first makes every comparison below it false, the second
    # asks for a step that violates the α > 0 contract, and silently ignoring either would hand
    # back exactly the unbounded step the caller was trying to rule out.
    αmax > zero(T) || invalid_αmax(αmax)
    αmax
end

# Behind a barrier for the same reason the reporting sites are: `caller_αmax` is specialized on the
# parameter type, and the message must not be compiled once per caller.
@noinline invalid_αmax(αmax) = throw(ArgumentError(
    "params.αmax = $(αmax) is not a usable ceiling on the step length; it has to be positive " *
    "(`Inf` is allowed and means the method's own ceiling stands). It bounds the α a line search " *
    "may return — see `linesearch_αmax`; omit the field entirely to say the same thing."))

"""
    change_precision(T, method::LinesearchMethod)

Return a copy of the [`LinesearchMethod`](@ref) `method` with its numeric fields
converted to the element type `T`.

This is an internal helper used when constructing a [`Linesearch`](@ref): the
method's precision is adapted to the working precision `T`.  It replaces a former
misuse of `Base.convert` (which was ambiguous with `Base` and violated the
`convert` contract by returning a differently-typed object).
"""
change_precision(::Type{T}, method::LinesearchMethod{T}) where {T} = method
change_precision(::Type, method::LinesearchMethod) =
    error("change_precision not implemented for $(typeof(method)).")


"""
    Linesearch

A `struct` that stores a [`LinesearchProblem`](@ref), [`LinesearchMethod`](@ref) and [`Options`](@ref).

# Keys

- `problem::`[`LinesearchProblem`](@ref)
- `method::`[`LinesearchMethod`](@ref)
- `config::`[`Options`](@ref)

# Constructors

The following constructors can be used:

```julia
Linesearch{T}(problem, method, config)
Linesearch(problem, method=Static(); kwargs...)
Linesearch(problem, method, config::Options)
```
"""
struct Linesearch{T,MET<:LinesearchMethod{T},PT<:LinesearchProblem{T},OPT<:Options{T}}
    problem::PT
    method::MET
    config::OPT
    Linesearch{T}(problem, method, config) where {T} = new{T,typeof(method),typeof(problem),typeof(config)}(problem, method, config)
end

Linesearch(problem::LinesearchProblem{T}, method::LinesearchMethod=Static(); options_kwargs...) where {T} =
    Linesearch{T}(problem, change_precision(T, method), Options(T; options_kwargs...))

Linesearch(problem::LinesearchProblem{T}, method::LinesearchMethod, config::Options{T}) where {T} =
    Linesearch{T}(problem, change_precision(T, method), config)

@doc raw"""
    with_config(ls, config)

Return a [`Linesearch`](@ref) with the [`LinesearchProblem`](@ref) and the
[`LinesearchMethod`](@ref) of `ls`, but with the [`Options`](@ref) `config`.

This is how a [`NonlinearSolver`](@ref) makes its line search share *its* options. A
`Linesearch` built by `Linesearch(problem, method)` carries an `Options` of its own,
constructed from nothing but defaults — so `verbosity` and `linesearch_max_iterations` would
be configured twice, and `verbosity = 0` on the solver would not silence the line search.

`Linesearch` is an immutable three-field wrapper, so rebuilding it is cheap: the problem (and
hence its closures and scratch buffers) and the method are shared, not copied.

The `Options` element type is pinned to the `Linesearch` element type, so a mismatched
`config` raises a `MethodError` rather than silently producing a broken object — the same
guarantee the three-argument [`Linesearch`](@ref) constructor gives.
"""
with_config(ls::Linesearch{T}, config::Options{T}) where {T} =
    Linesearch(problem(ls), method(ls), config)

function solve(prob::LinesearchProblem{T}, method::LinesearchMethod, α, params=NullParameters(), config::Options{T}=Options(T)) where {T}
    solve(Linesearch(prob, method, config), T(α), params)
end

problem(s::Linesearch) = s.problem
config(s::Linesearch) = s.config
method(s::Linesearch) = s.method


"""
    solve(linesearch, α, params=NullParameters())

Solve the [`LinesearchProblem`](@ref) (contained in [`Linesearch`](@ref)) starting at `α`,
report the outcome through [`linesearch_warnings`](@ref) and return the step length.

The argument `params` needs to be of an appropriate form expected by the respective [`LinesearchProblem`](@ref).

Use [`solve_with_status`](@ref) to obtain the [`LinesearchStatus`](@ref) instead: a caller that
has to tell "I found a decreasing step" from "the merit is at its round-off floor and nothing
can decrease it" cannot do so from the step length alone — and it is the call that emits no
messages, which is what a *program* wants (see [`record_linesearch!`](@ref)).

See [`linesearch_problem`](@ref).

# Implementation

This is *derived*, and it is the only definition: a [`LinesearchMethod`](@ref) implements
[`solve_with_status`](@ref), and `solve` is that plus the report. It used to be the other way
round — every method defined this same three-line body, and a method that defined only `solve`
got `solve_with_status` from a fallback that called it. That fallback made the layering of the
contract unenforceable: a third-party method reached through `solve` emits its messages from
wherever it is called, including from inside every iteration of a [`NonlinearSolver`](@ref),
which is precisely what a program must not see. With the direction reversed, there is no path
by which the package calls a method's `solve` during a solve, so the guarantee holds by
construction rather than by convention.

`α` is converted to the element type of the `Linesearch` here, as the problem-taking method
above does, so `solve(ls, 1)` on a `Linesearch{Float64}` means what it looks like it means.
"""
function solve(ls::Linesearch{T}, α, params=NullParameters()) where {T}
    status = solve_with_status(ls, T(α), params)
    linesearch_warnings(status, ls, params)
    steplength(status)
end
