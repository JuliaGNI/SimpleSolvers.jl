"""
    OptimizerStatus

Stores residuals (relative and absolute) and various convergence properties.

See [`OptimizerResult`](@ref).
"""
mutable struct OptimizerStatus{XT,YT}
    i::Int  # iteration number

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

OptimizerStatus{XT,YT}() where {XT,YT} = OptimizerStatus{XT,YT}(
        0, XT(NaN), XT(NaN), YT(NaN), YT(NaN), XT(NaN), XT(NaN), YT(NaN), YT(NaN),
        false, false, false, false, true, true, true)

OptimizerStatus{T}() where {T} = OptimizerStatus{T,T}()

OptimizerStatus(::AbstractArray{T₁}, ::AbstractArray{T₂}) where {T₁, T₂} = OptimizerStatus{T₁, T₂}()

iterations(status::OptimizerStatus) = status.i
x_abschange(status::OptimizerStatus) = status.rxₐ
x_relchange(status::OptimizerStatus) = status.rxᵣ
f_abschange(status::OptimizerStatus) = status.rfₐ
f_relchange(status::OptimizerStatus) = status.rfᵣ
f_change(status::OptimizerStatus) = status.Δf
f_change_approx(status::OptimizerStatus) = status.Δf̃
g_abschange(status::OptimizerStatus) = status.rgₐ
g_residual(status::OptimizerStatus) = status.rg

"""
    clear!(obj)

Similar to [`initialize!`](@ref).
"""
function clear!(status::OptimizerStatus{XT,YT}) where {XT,YT}
    status.i = 0

    status.rxₐ = XT(NaN)
    status.rxᵣ = XT(NaN)
    status.rfₐ = YT(NaN)
    status.rfᵣ = YT(NaN)
    status.rgₐ = YT(NaN)
    status.rg  = XT(NaN)

    status.Δf = YT(NaN)
    status.Δf̃ = YT(NaN)

    status.x_converged = false
    status.f_converged = false
    status.g_converged = false
    status.f_increased = false

    status.x_isnan = true
    status.f_isnan = true
    status.g_isnan = true

    status
end

function initialize!(status::OptimizerStatus{XT,YT}, ::AbstractVector{XT}) where {XT, YT}
    clear!(status)
end


"""
    residual!(status, x, x̄, f, f̄, g, ḡ)

Compute the residual based on previous iterates (`x̄`, `f̄`, `ḡ`) and current iterates (`x`, `f`, `g`).

Also see [`assess_convergence!`](@ref) and [`meets_stopping_criteria`](@ref).
"""
function residual!(status::OS, x::XT, x̄::XT, f::FT, f̄::FT, g::GT, ḡ::GT)::OS where {OS <: OptimizerStatus, XT, FT, GT}
    Δx = x - x̄
    status.rxₐ = norm(Δx)
    status.rxᵣ = status.rxₐ / norm(x)

    status.Δf  = f - f̄
    status.Δf̃ = ḡ ⋅ Δx

    status.rfₐ = norm(status.Δf)
    status.rfᵣ = status.rfₐ / norm(f)

    Δg = g - ḡ
    status.rgₐ = norm(Δg)
    status.rg  = norm(g)
    
    status.f_increased = abs(f) > abs(f̄)

    status.x_isnan = any(isnan, x)
    status.f_isnan = any(isnan, f)
    status.g_isnan = any(isnan, g)

    status
end

function Base.show(io::IO, s::OptimizerStatus)

    @printf io "\n"
    @printf io " * Iterations\n"
    @printf io "\n"
    @printf io "    n = %i\n" iterations(s)
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

function print_status(status::OptimizerStatus, config::Options)
    if (verbosity(config) ≥ 1 && !(assess_convergence!(status, config) && status.i ≤ config.max_iterations)) ||
        verbosity(config) > 1
        println(status)
    end
end

"""
    increase_iteration_number!(status)

Increase the iteration number of a `status`[`OptimizerStatus`](@ref). See [`increase_iteration_number!(::NonlinearSolverStatus)`](@ref).
"""
increase_iteration_number!(status::OptimizerStatus) = status.i += 1

isconverged(status::OptimizerStatus) = status.x_converged || status.f_converged || status.g_converged

"""
    assess_convergence!(status, config)

Checks if the optimizer converged.
"""
function assess_convergence!(status::OptimizerStatus, config::Options)
    x_converged = x_abschange(status) ≤ x_abstol(config) ||
                  x_relchange(status) ≤ x_reltol(config)
    
    f_converged = f_abschange(status) ≤ f_abstol(config) ||
                  f_relchange(status) ≤ f_reltol(config)
    
    f_converged_strong = f_change(status) ≤ f_mindec(config) * f_change_approx(status)

    g_converged = g_residual(status) ≤ g_restol(config)
    
    status.x_converged = x_converged
    status.f_converged = f_converged && f_converged_strong
    status.g_converged = g_converged

    isconverged(status)
end

@doc raw"""
    meets_stopping_criteria(status, config)

Check if the optimizer has converged.

# Implementation

`meets_stopping_criteria` first calls [`assess_convergence!`](@ref) and then checks if one of the following is true:
- `converged` (the output of [`assess_convergence!`](@ref)) is `true` and `status.i` ``\geq`` `config.min_iterations`,
- if `config.allow_f_increases` is `false`: `status.f_increased` is `true`,
- `status.i` ``\geq`` `config.max_iterations`,
- `status.rxₐ` ``>`` `config.x_abstol_break`
- `status.rxᵣ` ``>`` `config.x_reltol_break`
- `status.rfₐ` ``>`` `config.f_abstol_break`
- `status.rfᵣ` ``>`` `config.f_reltol_break`
- `status.rg`  ``>`` `config.g_restol_break`
- `status.x_isnan`
- `status.f_isnan`
- `status.g_isnan`
"""
function meets_stopping_criteria(status::OptimizerStatus, config::Options)
    converged = assess_convergence!(status, config)

    if status.x_isnan || status.f_isnan || status.g_isnan
        error("x, f or g in the OptimizerStatus you provided are NaNs.")
    end

    ( converged && status.i ≥ config.min_iterations ) ||
    ( status.f_increased && !config.allow_f_increases ) ||
      status.i ≥ config.max_iterations ||
      status.rxₐ > config.x_abstol_break ||
      status.rxᵣ > config.x_reltol_break ||
      status.rfₐ > config.f_abstol_break ||
      status.rfᵣ > config.f_reltol_break ||
      status.rg  > config.g_restol_break
end

function warn_iteration_number(status::OptimizerStatus, config::Options)
    if config.warn_iterations > 0 && status.i ≥ config.warn_iterations
        println("WARNING: Optimizer took ", status.i, " iterations.")
    end
end
