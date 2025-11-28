"""
    OptimizerStatus

Contains residuals (relative and absolute) and various convergence properties.

See [`OptimizerResult`](@ref).
"""
struct OptimizerStatus{XT,YT}
    rxₐ::XT  # absolute change in x
    rxᵣ::XT  # relative change in x
    rfₐ::YT  # absolute change in f
    rfᵣ::YT  # relative change in f
    rgₐ::YT  # absolute change in g
    rg::XT   # residual of g

    Δf::YT    # change of function
    Δf̃::YT

    x_converged::Bool
    f_converged::Bool
    g_converged::Bool
    f_increased::Bool

    x_isnan::Bool
    f_isnan::Bool
    g_isnan::Bool
end

x_abschange(status::OptimizerStatus) = status.rxₐ
x_relchange(status::OptimizerStatus) = status.rxᵣ
f_abschange(status::OptimizerStatus) = status.rfₐ
f_relchange(status::OptimizerStatus) = status.rfᵣ
f_change(status::OptimizerStatus) = status.Δf
f_change_approx(status::OptimizerStatus) = status.Δf̃
g_abschange(status::OptimizerStatus) = status.rgₐ
g_residual(status::OptimizerStatus) = status.rg

"""
    residual!(status, state, cache, f)

Compute the residual based on previous iterates (`x̄`, `f̄`, `ḡ`) (stored in e.g. [`NewtonOptimizerState`](@ref)) and current iterates (`x`, `f`, `g`) (partly stored in e.g. [`NewtonOptimizerCache`](@ref)).

Also [`meets_stopping_criteria`](@ref).
"""
function OptimizerStatus(state::OST, cache::OCT, f::T; config::Options) where {T, OST <: OptimizerState, OCT <: OptimizerCache{T}}
    Δx = cache.x - state.x̄
    rxₐ = norm(Δx)
    rxᵣ = rxₐ / norm(cache.x)

    Δf  = f - state.f̄
    Δf̃ = state.ḡ ⋅ Δx

    rfₐ = norm(Δf)
    rfᵣ = rfₐ / norm(f)

    Δg = cache.g - state.ḡ
    rgₐ = norm(Δg)
    rg  = norm(cache.g)
    
    f_increased = abs(f) > abs(state.f̄)

    x_isnan = any(isnan, cache.x)
    f_isnan = any(isnan, f)
    g_isnan = any(isnan, cache.g)

    _status = OptimizerStatus(rxₐ, rxᵣ, rfₐ, rfᵣ, rgₐ, rg, Δf, Δf̃, false, false, false, f_increased, x_isnan, f_isnan, g_isnan)

    (x_converged, f_converged, f_converged_strong, g_converged) = convergence_measures(_status, config)

    OptimizerStatus(rxₐ, rxᵣ, rfₐ, rfᵣ, rgₐ, rg, Δf, Δf̃, x_converged, f_converged, g_converged, f_increased, x_isnan, f_isnan, g_isnan)
end

function Base.show(io::IO, s::OptimizerStatus)

    @printf io "\n"
    @printf io " * Convergence measures\n"
    @printf io "\n"
    @printf io "    |x - x'|               = %.2e\n"  x_abschange(s)
    @printf io "    |x - x'|/|x'|          = %.2e\n"  x_relchange(s)
    @printf io "    |f(x) - f(x')|         = %.2e\n"  f_abschange(s)
    @printf io "    |f(x) - f(x')|/|f(x')| = %.2e\n"  f_relchange(s)
    @printf io "    |g(x) - g(x')|         = %.2e\n"  g_abschange(s)
    @printf io "    |g(x)|                 = %.2e\n"  g_residual(s) 
    @printf io "\n"

end

isconverged(status::OptimizerStatus) = status.x_converged || status.f_converged || status.g_converged

"""
    convergence_measures(status, config)

Checks if the optimizer converged.
"""
function convergence_measures(status::OptimizerStatus, config::Options)
    x_converged = x_abschange(status) ≤ x_abstol(config) ||
                  x_relchange(status) ≤ x_reltol(config)
    
    f_converged = f_abschange(status) ≤ f_abstol(config) ||
                  f_relchange(status) ≤ f_reltol(config)
    
    f_converged_strong = f_change(status) ≤ f_mindec(config) * f_change_approx(status)

    g_converged = g_residual(status) ≤ g_restol(config)
    
    (x_converged, f_converged, f_converged_strong, g_converged)
end

@doc raw"""
    meets_stopping_criteria(status, config, iterations)

Check if the optimizer has converged.

# Implementation

`meets_stopping_criteria` checks if one of the following is true:
- `converged` (the output of [`assess_convergence!`](@ref)) is `true` and `iterations` ``\geq`` `config.min_iterations`,
- if `config.allow_f_increases` is `false`: `status.f_increased` is `true`,
- `iterations` ``\geq`` `config.max_iterations`,
- `status.rxₐ` ``>`` `config.x_abstol_break`
- `status.rxᵣ` ``>`` `config.x_reltol_break`
- `status.rfₐ` ``>`` `config.f_abstol_break`
- `status.rfᵣ` ``>`` `config.f_reltol_break`
- `status.rg`  ``>`` `config.g_restol_break`
- `status.x_isnan`
- `status.f_isnan`
- `status.g_isnan`
"""
function meets_stopping_criteria(status::OptimizerStatus, config::Options, iterations::Integer)
    converged = isconverged(status)

    if iterations ≥ 1 && (status.x_isnan || status.f_isnan || status.g_isnan)
        @error "x, f or g in the OptimizerStatus you provided are NaNs."
    end

    ( converged && iterations ≥ config.min_iterations ) ||
    ( status.f_increased && !config.allow_f_increases ) ||
      iterations ≥ config.max_iterations ||
      status.rxₐ > config.x_abstol_break ||
      status.rxᵣ > config.x_reltol_break ||
      status.rfₐ > config.f_abstol_break ||
      status.rfᵣ > config.f_reltol_break ||
      status.rg  > config.g_restol_break
end
