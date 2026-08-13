"""
    Picard <: NonlinearSolverMethod

See [`PicardSolver`](@ref).
"""
struct Picard <: NonlinearSolverMethod end

const PicardSolver{T} = NonlinearSolver{T,Picard}

function PicardSolver(x::AT, nlp::NLST, linesearch::LiSeT, cache::CT, config::Options{T}; jacobian) where {T,AT<:AbstractVector{T},NLST,LiSeT<:Linesearch{T},CT}
    NonlinearSolver(x, nlp, NoLinearProblem(), NoLinearSolver(), linesearch, cache, config; jacobian=jacobian, method=Picard())
end

# See the corresponding `NewtonSolver` method: the line search is rebuilt on the solver's
# `Options` so that solver and line search cannot be configured inconsistently.
function PicardSolver(x::AT, nlp::NLST, linesearch::LiSeT, cache::CT; jacobian, options_kwargs...) where {T,AT<:AbstractVector{T},NLST,LiSeT<:Linesearch{T},CT}
    config = Options(T; options_kwargs...)
    PicardSolver(x, nlp, with_config(linesearch, config), cache, config; jacobian=jacobian)
end

"""
    PicardSolver(x, F)

# Arguments
- `x`: the initial guess for the solution.
- `F`: the nonlinear function to solve.
- `y`

# Keywords
- `DF!`: the Jacobian of `F`,
- `jacobian`: the Jacobian of `F`, defaults to [`JacobianAutodiff`](@ref),
- `options_kwargs`: see [`Options`](@ref).

Note that the Picard [`solver_step!`](@ref) is a residual-safeguarded *fixed-point
iteration* and uses no line search, so — unlike the other solvers — no `linesearch`
keyword is accepted (passing one is an error rather than being silently ignored).

# Examples

```jldoctest; setup = :(using SimpleSolvers)
F(y, x, params) = y .= sin.(x) .^ 2
x = zeros(2)
y = similar(x)

s = PicardSolver(x, F, y)
state = SolverState(s)

solve!(x, s, state)

# output

2-element Vector{Float64}:
 0.0
 0.0
```
"""
function PicardSolver(x::AT, F::Callable, y::AT; (DF!)=missing, kwargs...) where {T,AT<:AbstractVector{T}}
    PicardSolver(x, NonlinearProblem(F, DF!, x, y), y; kwargs...)
end

"""
    PicardSolver(x, nlp::NonlinearProblem, y = zero(x))

Build a [`PicardSolver`](@ref) for the [`NonlinearProblem`](@ref) `nlp` with the initial
guess `x`. See [`NewtonSolver(::AbstractVector{T}, ::NonlinearProblem, ::AbstractVector{T}) where {T}`](@ref)
for the rôle of `y`; as above, no `linesearch` keyword is accepted.

# Keywords
- `jacobian`: see [`resolve_jacobian`](@ref),
- `options_kwargs`: see [`Options`](@ref).
"""
function PicardSolver(x::AbstractVector{T}, nlp::NonlinearProblem, y::AbstractVector{T}=zero(x); jacobian=missing, options_kwargs...) where {T}
    config = Options(T; options_kwargs...)
    jacobian = resolve_jacobian(nlp.F, nlp.J, jacobian, x, y)
    cache = NonlinearSolverCache(x, y)
    # The Picard `solver_step!` never consults a line search; the (structurally
    # mandatory) `linesearch` field is filled with a trivial `Static` step.  A
    # `linesearch` keyword is deliberately not accepted — it would be silently
    # ignored (any stray keyword falls through to `Options` and errors there).
    ls = Linesearch(linesearch_problem(nlp, jacobian, cache), Static(one(T)), config)
    PicardSolver(x, nlp, ls, cache, config; jacobian=jacobian)
end

function PicardSolver(x::AT, y::AT; F=missing, kwargs...) where {T,AT<:AbstractVector{T}}
    !ismissing(F) || error("You have to provide an F.")
    PicardSolver(x, F, y; kwargs...)
end

NonlinearSolver(::Picard, x...; kwargs...) = PicardSolver(x...; kwargs...)

function direction!(d::AbstractVector{T}, x::AbstractVector{T}, it::PicardSolver{T}, params) where {T}
    value!(d, nonlinearproblem(it), x, params)
    d .*= -1
end

function direction!(it::PicardSolver, x::AbstractVector, params)
    direction!(direction(cache(it)), x, it, params)
end

# The `iteration` and `stalled` arguments are part of the shared `direction!` interface; a
# Picard step has no Jacobian to refactorize, so both are ignored.
direction!(it::PicardSolver, x::AbstractVector, params, iteration; stalled::Bool=false) = direction!(it, x, params)

"Backtracking shrink factor used by the [`PicardSolver`](@ref) residual safeguard."
const DEFAULT_PICARD_BACKTRACKING_p = 0.5

@doc raw"""
    solver_step!(x, s::PicardSolver, state, params)

Take one *fixed-point* (Picard) step ``x \gets x + \alpha d`` with the residual
direction ``d = -F(x)`` (see [`direction!`](@ref)).

Unlike a Newton/Gauss-Newton step, the Picard direction ``d = -F(x)`` is **not**
in general a descent direction for the merit ``\varphi = \|F\|^2``, so applying the
derivative-based (Wolfe) line search used by the other [`NonlinearSolver`](@ref)s is
inappropriate (a directional derivative that is not negative makes the sufficient-
decrease/curvature tests meaningless).

Instead the step is *damped* by a **residual-monotonicity backtracking**: starting
from the full fixed-point step ``\alpha = 1`` the step is halved until the residual
norm does not increase, ``\|F(x + \alpha d)\| \le \|F(x)\|``.  This safeguard uses
only function values and makes no descent assumption.  If no positive ``\alpha``
reduces the residual (the fixed-point map is locally expanding), the smallest trial
step is taken and the convergence test — which requires a small
*residual*, not merely a small step — correctly reports non-convergence instead of
a false positive.
"""
function solver_step!(x::AbstractVector{T}, s::PicardSolver{T}, state::NonlinearSolverState{T}, params) where {T}
    direction!(s, x, params, iteration_number(state))
    # The Picard direction *is* the residual, so this rejects a non-finite `F(x)` at the current
    # iterate. As in the generic step, damping cannot rescue a non-finite direction.
    all(isfinite, direction(cache(s))) || throw(NonlinearSolverException("non-finite direction vector"))

    # NaN recovery on the residual direction (mirrors the generic solver step); leaves
    # the α = 1 trial residual in `value(cache(s))`, reused by the safeguard below.
    nan_recovery!(s, x, params)

    # Damped fixed-point step with a residual-monotonicity safeguard: starting from the
    # full step α = 1 — whose residual ‖F(x + d)‖ is already in `value(cache)` from the NaN
    # loop above, so it is *not* recomputed here — halve α until the residual no longer
    # increases (‖F(x + αd)‖ ≤ ‖F(x)‖) or the step underflows (α ≤ eps). The committed
    # iterate is always the last one actually *evaluated* (`solution(cache)`), so its
    # residual was checked; if the map is locally expanding the smallest trial step is taken
    # and the convergence test correctly reports non-convergence. The loop is bounded by the
    # step underflow (not by `max_iterations`, which caps the outer iteration).
    r₀ = l2norm(value(state))                       # ‖F(x)‖ at the current iterate
    p = T(DEFAULT_PICARD_BACKTRACKING_p)
    α = one(T)
    while !(l2norm(value(cache(s))) ≤ r₀) && α > eps(T)
        α *= p
        compute_new_iterate!(solution(cache(s)), x, α, direction(cache(s)))
        value!(value(cache(s)), nonlinearproblem(s), solution(cache(s)), params)
    end
    # A trial iterate that is not finite must not be committed, even though it is the last one
    # evaluated: it is reachable when `nan_recovery!` exhausted `nan_max_iterations` without
    # escaping the region where `F` is undefined or overflows. Leaving `x` where it was makes
    # the step a frozen one, which `stalled_step` already diagnoses — the same choice
    # `DogLegSolver` makes for a rejected trial on radius underflow.
    all(isfinite, solution(cache(s))) && (x .= solution(cache(s)))

    x
end
