
mutable struct OptimizerStatus{XT,YT,VT,OT}
    config::Options{OT}

    i::Int  # iteration number

    rxₐ::XT  # absolute change in x
    rxᵣ::XT  # relative change in x
    rfₐ::YT  # absolute change in f
    rfᵣ::YT  # relative change in f
    rgₐ::XT  # absolute change in g
    rgᵣ::XT  # relative change in g
    rg::XT   # residual of g

    x̄::VT    # previous solution
    x::VT    # current solution
    δ::VT

    f̄::YT    # previous function
    f::YT    # current function

    ḡ::VT    # previous gradient
    g::VT    # current gradient
    γ::VT

    x_converged::Bool
    f_converged::Bool
    g_converged::Bool
    f_increased::Bool
end

OptimizerStatus{XT,YT,VT}(config::Options{OT}, n) where {XT,YT,VT,OT} = OptimizerStatus{XT,YT,VT,OT}(
    config,
    0, XT(NaN), XT(NaN), YT(NaN), YT(NaN), XT(NaN), XT(NaN), XT(NaN),
    zeros(XT,n), zeros(XT,n), zeros(XT,n),
    YT(NaN), YT(NaN),
    zeros(XT,n), zeros(XT,n), zeros(XT,n),
    false, false, false, false)

OptimizerStatus{T}(config, n) where {T} = OptimizerStatus{T,T,Vector{T}}(config,n)
OptimizerStatus(config, x, y, g) = OptimizerStatus{eltype(x),typeof(y),typeof(g)}(config, length(x))

solution(status::OptimizerStatus) = status.x

x_abschange(status::OptimizerStatus) = status.rxₐ
x_relchange(status::OptimizerStatus) = status.rxᵣ
f_abschange(status::OptimizerStatus) = status.rfₐ
f_relchange(status::OptimizerStatus) = status.rfᵣ
g_residual(status::OptimizerStatus) = status.rg

function clear!(status::OptimizerStatus{XT,YT}) where {XT,YT}
    status.i = 0

    status.rxₐ = XT(NaN)
    status.rxᵣ = XT(NaN)
    status.rfₐ = YT(NaN)
    status.rfᵣ = YT(NaN)
    status.rgₐ = XT(NaN)
    status.rgᵣ = XT(NaN)
    status.rg  = XT(NaN)

    status.x̄ .= XT(NaN)
    status.x .= XT(NaN)
    status.δ .= XT(NaN)
    status.f̄  = YT(NaN)
    status.f  = YT(NaN)
    status.ḡ .= XT(NaN)
    status.g .= XT(NaN)
    status.γ .= XT(NaN)

    status.x_converged = false
    status.f_converged = false
    status.g_converged = false
    status.f_increased = false
end

function Base.show(io::IO, s::OptimizerStatus)

    @printf io "\n"
    @printf io " * Convergence measures\n"
    @printf io "\n"
    @printf io "    |x - x'|               = %.2e %s %.1e\n"  x_abschange(s) x_abschange(s) ≤ x_abstol(s.config) ? "≤" : "≰" x_abstol(s.config)
    @printf io "    |x - x'|/|x'|          = %.2e %s %.1e\n"  x_relchange(s) x_relchange(s) ≤ x_reltol(s.config) ? "≤" : "≰" x_reltol(s.config)
    @printf io "    |f(x) - f(x')|         = %.2e %s %.1e\n"  f_abschange(s) f_abschange(s) ≤ f_abstol(s.config) ? "≤" : "≰" f_abstol(s.config)
    @printf io "    |f(x) - f(x')|/|f(x')| = %.2e %s %.1e\n"  f_relchange(s) f_relchange(s) ≤ f_reltol(s.config) ? "≤" : "≰" f_reltol(s.config)
    @printf io "    |g(x)|                 = %.2e %s %.1e\n"  g_residual(s)  g_residual(s)  ≤ g_reltol(s.config) ? "≤" : "≰" g_reltol(s.config)
    @printf io "\n"

    @printf io " * Candidate solution\n"
    @printf io "\n"
    length(s.x) > 10 || @printf io  "    Final solution value:     [%s]\n" join([@sprintf "%e" x for x in s.x], ", ")
    @printf io "    Final objective value:     %e\n" s.f
    @printf io "\n"

end
function print_status(status::OptimizerStatus, config::Options)
    if (config.verbosity ≥ 1 && !(assess_convergence!(status, config) && status.i ≤ config.max_iterations)) ||
        config.verbosity > 1
        println(status)
    end
end

increase_iteration_number!(status::OptimizerStatus) = status.i += 1

isconverged(status::OptimizerStatus) = status.x_converged || status.f_converged || status.g_converged

function assess_convergence!(status::OptimizerStatus, config::Options)
    x_converged = status.rxₐ ≤ config.x_abstol ||
                  status.rxᵣ ≤ config.x_reltol
    
    f_converged = status.rfₐ ≤ config.f_abstol ||
                  status.rfᵣ ≤ config.f_reltol

    g_converged = status.rgₐ ≤ config.g_abstol ||
                  status.rgᵣ ≤ config.g_reltol
    
    status.x_converged = x_converged
    status.f_converged = f_converged
    status.g_converged = g_converged

    status.f_increased = abs(status.f) > abs(status.f̄)

    return isconverged(status)
end

function meets_stopping_criteria(status::OptimizerStatus, config::Options)
    assess_convergence!(status, config)

    ( isconverged(status) && status.i ≥ config.min_iterations ) ||
    ( status.f_increased && !config.allow_f_increases ) ||
      status.i ≥ config.max_iterations ||
      status.rxₐ > config.x_abstol_break ||
      status.rxᵣ > config.x_reltol_break ||
      status.rfₐ > config.f_abstol_break ||
      status.rfᵣ > config.f_reltol_break ||
      status.rgₐ > config.g_abstol_break ||
      status.rgᵣ > config.g_reltol_break ||
      any(isnan, status.x) ||
      any(isnan, status.f) ||
      any(isnan, status.g)
end


# function check_solver_status(status::OptimizerStatus, config::Options)
#     if any(isnan, status.x) || isnan(status.f) || any(isnan, status.g)
#         throw(NonlinearSolverException("Detected NaN"))
#     end

#     if status.rfₐ > config.f_abstol_break
#         throw(NonlinearSolverException("Absolute error ($(status.rfₐ)) larger than allowed ($(config.f_abtol_break))"))
#     end

#     if status.rfᵣ > config.f_reltol_break
#         throw(NonlinearSolverException("Relative error ($(status.rfᵣ)) larger than allowed ($(config.f_reltol_break))"))
#     end
# end

# function get_solver_status!(status::OptimizerStatus, params::NonlinearSolverParameters, status_dict::Dict)
#     status_dict[:niter] = status.i
#     status_dict[:xatol] = status.rxₐ
#     status_dict[:xrtol] = status.rxᵣ
#     status_dict[:yatol] = status.rfₐ
#     status_dict[:yrtol] = status.rfᵣ
#     status_dict[:gatol] = status.rgₐ
#     status_dict[:grtol] = status.rgᵣ
#     status_dict[:converged] = assess_convergence(status, params)
#     return status_dict
# end

# get_solver_status!(solver::OptimizerStatus{T}, status_dict::Dict) where {T} =
#             get_solver_status!(status(solver), params(solver), status_dict)

# get_solver_status(solver::OptimizerStatus{T}) where {T} = get_solver_status!(solver,
#             Dict(:niter => 0,
#                  :xatol => zero(T),
#                  :xrtol => zero(T),
#                  :yatol => zero(T),
#                  :yrtol => zero(T),
#                  :gatol => zero(T),
#                  :grtol => zero(T),
#                  :converged => false)
#             )


function warn_iteration_number(status::OptimizerStatus, config::Options)
    if config.warn_iterations > 0 && status.i ≥ config.warn_iterations
        println("WARNING: Optimizer took ", status.i, " iterations.")
    end
end


function residual!(status::OptimizerStatus)
    status.rxₐ = norm(status.δ)
    status.rxᵣ = status.rxₐ / norm(status.x)

    status.rfₐ = norm(status.f̄ - status.f)
    status.rfᵣ = status.rfₐ / norm(status.f)

    status.rg  = norm(status.g)
    status.rgₐ = norm(status.γ)
    status.rgᵣ = status.rgₐ / norm(status.g)
end

function residual!(status::OptimizerStatus, x, f, g)
    status.x .= x
    status.f  = f
    status.g .= g

    status.δ .= status.x .- status.x̄
    status.γ .= status.g .- status.ḡ

    residual!(status)
end

function initialize!(status::OptimizerStatus, x, f, g)
    clear!(status)
    status.x̄ .= status.x .= x
    status.f̄  = status.f  = f
    status.ḡ .= status.g .= g
    status.δ .= 0
    status.γ .= 0
end

function next_iteration!(status::OptimizerStatus)
    increase_iteration_number!(status)
    status.x̄ .= status.x
    status.f̄  = status.f
    status.ḡ .= status.g
end
