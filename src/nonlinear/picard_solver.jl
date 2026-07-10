"""
    Picard <: NonlinearSolverMethod

See [`PicardSolver`](@ref).
"""
struct Picard <: NonlinearSolverMethod end

const PicardSolver{T} = NonlinearSolver{T,Picard}

function PicardSolver(x::AT, nlp::NLST, linesearch::LiSeT, cache::CT; jacobian, options_kwargs...) where {T,AT<:AbstractVector{T},NLST,LiSeT,CT}
    config = Options(T; options_kwargs...)
    NonlinearSolver(x, nlp, NoLinearProblem(), NoLinearSolver(), linesearch, cache, config; jacobian=jacobian, method=Picard())
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
function PicardSolver(x::AT, F::Callable, y::AT; (DF!)=missing, jacobian=JacobianAutodiff(F, x, y), kwargs...) where {T,AT<:AbstractVector{T}}
    nlp = NonlinearProblem(F, DF!, x, y)
    jacobian = ismissing(DF!) ? jacobian : JacobianFunction{T}(F, DF!)
    cache = NonlinearSolverCache(x, y)
    # The Picard `solver_step!` never consults a line search; the (structurally
    # mandatory) `linesearch` field is filled with a trivial `Static` step.  A
    # `linesearch` keyword is deliberately not accepted — it would be silently
    # ignored (any stray keyword falls through to `Options` and errors there).
    ls = Linesearch(linesearch_problem(nlp, jacobian, cache), Static(one(T)))
    PicardSolver(x, nlp, ls, cache; jacobian=jacobian, kwargs...)
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

direction!(it::PicardSolver, x::AbstractVector, params, iteration) = direction!(it, x, params)

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
    any(isnan, direction(cache(s))) && throw(NonlinearSolverException("NaN detected in direction vector"))

    # NaN recovery on the residual direction (mirrors the generic solver step).
    for _ in 1:config(s).nan_max_iterations
        solution(cache(s)) .= x .+ direction(cache(s))
        value!(value(cache(s)), nonlinearproblem(s), solution(cache(s)), params)
        if any(isnan, value(cache(s)))
            (config(s).verbosity ≥ 2 && @warn "NaN detected in nonlinear solver. Reducing length of direction vector.")
            direction(cache(s)) .*= T(config(s).nan_factor)
        else
            break
        end
    end

    # Damped fixed-point step with a residual-monotonicity safeguard.
    r₀ = l2norm(value(state))                       # ‖F(x)‖ at the current iterate
    p = T(DEFAULT_PICARD_BACKTRACKING_p)
    α = one(T)
    for _ in 1:config(s).max_iterations
        compute_new_iterate!(solution(cache(s)), x, α, direction(cache(s)))
        value!(value(cache(s)), nonlinearproblem(s), solution(cache(s)), params)
        (l2norm(value(cache(s))) ≤ r₀ || α ≤ eps(T)) && break
        α *= p
    end
    compute_new_iterate!(x, α, direction(cache(s)))

    x
end
