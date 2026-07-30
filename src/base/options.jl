"""
    default_tolerance(T)

Determine the default tolerance for a specific data type. This is used in the constructor of [`Options`](@ref).

Compare this to [`default_precision`](@ref).

# Examples

```jldoctest; setup = :(using SimpleSolvers: default_tolerance)
julia> default_tolerance(Float64)
4.440892098500626e-16
```

```jldoctest; setup = :(using SimpleSolvers: default_tolerance)
julia> default_tolerance(Float32)
2.3841858f-7
```

```jldoctest; setup = :(using SimpleSolvers: default_tolerance)
julia> default_tolerance(Float16)
Float16(0.001953)
```
"""
function default_tolerance(::Type{T}) where {T<:AbstractFloat}
    2eps(T)
end

"""
    absolute_tolerance(T)

Determine the absolute tolerance for a specific data type. This is used in the constructor of [`Options`](@ref).

In comparison to [`default_tolerance`](@ref), this should return a very small number, close to zero (i.e. not just machine precision).

# Examples

```jldoctest; setup = :(using SimpleSolvers: absolute_tolerance)
julia> absolute_tolerance(Float64)
0.0
```

```jldoctest; setup = :(using SimpleSolvers: absolute_tolerance)
julia> absolute_tolerance(Float32)
0.0f0
```
"""
function absolute_tolerance(::Type{T}) where {T<:AbstractFloat}
    zero(T)
end

"""
    minimum_decrease_threshold(T)

The minimum value by which a function ``f`` should decrease during an iteration.

The default value of ``10^{-4}`` is often used in the literature [bierlaire2015optimization](@cite), [nocedal2006numerical](@cite).

# Examples

```jldoctest; setup = :(using SimpleSolvers: minimum_decrease_threshold)
julia> minimum_decrease_threshold(Float64)
0.0001
```

```jldoctest; setup = :(using SimpleSolvers: minimum_decrease_threshold)
julia> minimum_decrease_threshold(Float32)
0.0001f0
```
"""
function minimum_decrease_threshold(::Type{T}) where {T<:AbstractFloat}
    T(10)^-4
end

@doc raw"""
    linesearch_iterations(T)

Determine the default number of trial steps a line search may take, i.e. the default of the
`linesearch_max_iterations` field of [`Options`](@ref).

This is deliberately *not* the same quantity as `max_iterations`, which bounds the outer
nonlinear iteration (see [`meets_stopping_criteria`](@ref)). A one-dimensional search *inside
a single solver step* needs a budget on the order of the mantissa width, not thousands of
trials: a [`Backtracking`](@ref) ladder ``\alpha \gets p\alpha`` starting at ``\alpha_0 = 1``
reaches the negligible-step floor after ``\lceil-\log_2\varepsilon\rceil`` halvings (52 in
double precision, 24 in single), and a [`bisection`](@ref) needs the same count to exhaust the
mantissa. We take that count plus a small margin; everything beyond it can only produce
denormals.

Note that [`Quadratic`](@ref) and [`BierlaireQuadratic`](@ref) are *not* bounded by this: they
fit a quadratic rather than shrink a step, and keep their own
`max_number_of_quadratic_linesearch_iterations`.

Compare this to [`default_tolerance`](@ref) and [`absolute_tolerance`](@ref).

# Examples

```jldoctest; setup = :(using SimpleSolvers: linesearch_iterations)
julia> linesearch_iterations(Float64)
60
```

```jldoctest; setup = :(using SimpleSolvers: linesearch_iterations)
julia> linesearch_iterations(Float32)
31
```
"""
function linesearch_iterations(::Type{T}) where {T<:AbstractFloat}
    ceil(Int, -log2(eps(T))) + 8
end

const ALLOW_F_INCREASES::Bool = true
const MIN_ITERATIONS::Int = 0
const MAX_ITERATIONS::Int = 1_000
const WARN_ITERATIONS::Int = 1_000
const SHOW_TRACE::Bool = false
const STORE_TRACE::Bool = false
const EXTENDED_TRACE::Bool = false
const SHOW_EVERY::Int = 1
const VERBOSITY::Int = 1

const NAN_MAX_ITERATIONS = 10
const NAN_FACTOR = 0.5
const REGULARIZATION_FACTOR = 0

@doc raw"""
    const MAX_STALLS

The default number of *consecutive* stalled steps after which a [`NonlinearSolver`](@ref)
gives up; the default of the `max_stalls` field of [`Options`](@ref). Its value is """ *
                     """$(MAX_STALLS)""" * raw""".

A step is stalled when it does not move the iterate while the residual is not small (see
[`stalled_step`](@ref)), i.e. when the merit ``\|F\|^2`` cannot be reduced along the current
direction. One stalled step is not conclusive: a quasi-Newton solver refreshes its
[`Jacobian`](@ref) on a later step (see [`maybe_refactorize!`](@ref)) and the
[`DogLegSolver`](@ref) resets a collapsed trust-region radius on the step *after* a rejected
one, so both can still make progress after a frozen step. Two in a row are conclusive.

Set `max_stalls = typemax(Int)` to restore the previous behaviour of running all the way to
`max_iterations`.
"""
const MAX_STALLS::Int = 2

"""
    Options

# Examples

```jldoctest; setup = :(using SimpleSolvers)
Options()

# output

                x_abstol = 4.440892098500626e-16
                x_reltol = 4.440892098500626e-16
                x_suctol = 4.440892098500626e-16
                f_abstol = 0.0
                f_reltol = 1.4901161193847656e-8
                f_suctol = 4.440892098500626e-16
                f_mindec = 0.0001
          f_abstol_break = Inf
       allow_f_increases = true
          min_iterations = 0
          max_iterations = 1000
         warn_iterations = 1000
linesearch_max_iterations = 60
              max_stalls = 2
              show_trace = false
             store_trace = false
          extended_trace = false
              show_every = 1
               verbosity = 1
      nan_max_iterations = 10
              nan_factor = 0.5
   regularization_factor = 0.0
   dogleg_radius_initial = 1.0
    dogleg_radius_shrink = 0.25
    dogleg_radius_expand = 2.0
       dogleg_radius_max = 100.0

```

!!! info
    The tolerance constants (`x_abstol` through `f_suctol`) default to values derived from
    [`default_tolerance`](@ref) and [`absolute_tolerance`](@ref), except `f_reltol`, which
    defaults to `√eps(T)`: it is the relative residual tolerance used by
    [`assess_convergence`](@ref) — the residual is small when `rfₐ ≤ f_abstol + f_reltol·‖F(x₀)‖`,
    i.e. the absolute tolerance is `f_abstol` and the relative tolerance is `f_reltol`.

!!! info
    `dogleg_radius_initial`, `dogleg_radius_shrink`, `dogleg_radius_expand` and
    `dogleg_radius_max` are the trust-region parameters for the [`DogLegSolver`](@ref):
    the initial and maximum radius (``\\Delta_0`` and ``\\hat\\Delta`` in
    [nocedal2006numerical; Alg. 4.1](@cite)) and the factors by which the radius is
    shrunk on a poor step / expanded on a very good boundary step. They default to
    [`DOGLEG_Δ_INITIAL`](@ref), [`DOGLEG_Δ_SHRINK`](@ref), [`DOGLEG_Δ_EXPAND`](@ref) and
    [`DOGLEG_Δ_MAX`](@ref), and are ignored by the other solvers.

!!! info "`max_iterations` versus `linesearch_max_iterations`"
    `max_iterations` bounds the **outer** nonlinear iteration (see
    [`meets_stopping_criteria`](@ref)); `linesearch_max_iterations` bounds the **inner**,
    one-dimensional line search taken within a single solver step — the
    [`Backtracking`](@ref) ladder, the [`StrongWolfe`](@ref) bracketing and zoom phases, and
    [`bisection`](@ref). These used to be the same field, which meant that capping the solver
    at `max_iterations = 50` silently also capped the ladder, and that the default of 1000 was
    applied to a ladder which can never need more than ``\\lceil-\\log_2\\varepsilon\\rceil``
    trials. See [`linesearch_iterations`](@ref).

!!! warning "Choosing `f_abstol`"
    `f_abstol` is an *absolute* target for ``\\|F(x)\\|``, and the default `0` (see
    [`absolute_tolerance`](@ref)) is never met by a nonzero residual: the absolute branch of
    [`assess_convergence`](@ref) is switched *off* by default and convergence is decided
    entirely by the relative (`f_reltol`) and successive (`x_suctol`, `f_suctol`) branches.

    Conversely, an `f_abstol` **below the round-off floor of your own residual** — the
    cancellation level of the terms `F` sums internally, which the solver cannot see — is
    *unsatisfiable*. The iteration then reaches that floor, stops making progress, and is
    reported as stagnated (see `max_stalls`, [`stalled_step`](@ref) and
    [`nonlinear_solver_warnings`](@ref)) rather than converged.

    Note that `f_reltol` does **not** rescue this case: the relative gate is anchored at the
    *initial* residual ``\\|F(x_0)\\|``, so an excellent initial guess makes it *tighter*, not
    looser. If the stagnation warning reports an achieved `rfₐ` near your `f_abstol`, raise
    `f_abstol` above it — an order of magnitude of headroom is usual.

!!! info
    Also see [`meets_stopping_criteria`](@ref).
"""
struct Options{T}
    x_abstol::T
    x_reltol::T
    x_suctol::T
    f_abstol::T
    f_reltol::T
    f_suctol::T
    f_mindec::T
    f_abstol_break::T
    allow_f_increases::Bool
    min_iterations::Int
    max_iterations::Int
    warn_iterations::Int
    linesearch_max_iterations::Int
    max_stalls::Int
    show_trace::Bool
    store_trace::Bool
    extended_trace::Bool
    show_every::Int
    verbosity::Int
    nan_max_iterations::Int
    nan_factor::T
    regularization_factor::T
    dogleg_radius_initial::T
    dogleg_radius_shrink::T
    dogleg_radius_expand::T
    dogleg_radius_max::T
end

function Options(T=Float64;
    x_abstol::Real=default_tolerance(T),
    x_reltol::Real=default_tolerance(T),
    x_suctol::Real=default_tolerance(T),
    f_abstol::Real=absolute_tolerance(T),
    f_reltol::Real=(√(eps(T))),
    f_suctol::Real=default_tolerance(T),
    f_mindec::Real=minimum_decrease_threshold(T),
    f_abstol_break::Real=T(Inf),
    allow_f_increases::Bool=ALLOW_F_INCREASES,
    min_iterations::Integer=MIN_ITERATIONS,
    max_iterations::Integer=MAX_ITERATIONS,
    warn_iterations::Integer=WARN_ITERATIONS,
    linesearch_max_iterations::Integer=linesearch_iterations(T),
    max_stalls::Integer=MAX_STALLS,
    show_trace::Bool=SHOW_TRACE,
    store_trace::Bool=STORE_TRACE,
    extended_trace::Bool=EXTENDED_TRACE,
    show_every::Integer=SHOW_EVERY,
    verbosity::Integer=VERBOSITY,
    nan_max_iterations::Integer=NAN_MAX_ITERATIONS,
    nan_factor::Real=NAN_FACTOR,
    regularization_factor::Real=T(REGULARIZATION_FACTOR),
    dogleg_radius_initial::Real=T(DOGLEG_Δ_INITIAL),
    dogleg_radius_shrink::Real=T(DOGLEG_Δ_SHRINK),
    dogleg_radius_expand::Real=T(DOGLEG_Δ_EXPAND),
    dogleg_radius_max::Real=T(DOGLEG_Δ_MAX),
)

    show_every = show_every > 0 ? show_every : 1

    Options{T}(promote(x_abstol,
            x_reltol,
            x_suctol,
            f_abstol,
            f_reltol,
            f_suctol,
            f_mindec,
            f_abstol_break)...,
        allow_f_increases,
        min_iterations,
        max_iterations,
        warn_iterations,
        linesearch_max_iterations,
        max_stalls,
        show_trace,
        store_trace,
        extended_trace,
        show_every,
        verbosity,
        nan_max_iterations,
        nan_factor,
        regularization_factor,
        dogleg_radius_initial,
        dogleg_radius_shrink,
        dogleg_radius_expand,
        dogleg_radius_max,
    )
end

function Base.show(io::IO, o::SimpleSolvers.Options)
    for k in fieldnames(typeof(o))
        v = getfield(o, k)
        if v isa Nothing
            @printf io "%24s = %s\n" k "nothing"
        else
            @printf io "%24s = %s\n" k v
        end
    end
end

x_abstol(o::Options) = o.x_abstol
x_reltol(o::Options) = o.x_reltol
x_suctol(o::Options) = o.x_suctol
f_abstol(o::Options) = o.f_abstol
f_reltol(o::Options) = o.f_reltol
f_suctol(o::Options) = o.f_suctol
f_mindec(o::Options) = o.f_mindec

verbosity(o::Options) = o.verbosity

"""
    linesearch_max_iterations(o::Options)

The maximum number of trial steps a line search may take within a single solver step. See
[`linesearch_iterations`](@ref); *not* to be confused with `max_iterations`, which bounds the
outer nonlinear iteration.
"""
linesearch_max_iterations(o::Options) = o.linesearch_max_iterations

"""
    max_stalls(o::Options)

The number of consecutive stalled steps after which a [`NonlinearSolver`](@ref) gives up. See
[`MAX_STALLS`](@ref) and [`stalled_step`](@ref).
"""
max_stalls(o::Options) = o.max_stalls
