
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

function OptimizerStatus(x, y)
    XT = typeof(norm(x))
    YT = typeof(norm(y))
    OptimizerStatus{XT,YT}()
end

iterations(status::OptimizerStatus) = status.i
x_abschange(status::OptimizerStatus) = status.rxₐ
x_relchange(status::OptimizerStatus) = status.rxᵣ
f_abschange(status::OptimizerStatus) = status.rfₐ
f_relchange(status::OptimizerStatus) = status.rfᵣ
f_change(status::OptimizerStatus) = status.Δf
f_change_approx(status::OptimizerStatus) = status.Δf̃
g_abschange(status::OptimizerStatus) = status.rgₐ
g_residual(status::OptimizerStatus) = status.rg

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

increase_iteration_number!(status::OptimizerStatus) = status.i += 1

isconverged(status::OptimizerStatus) = status.x_converged || status.f_converged || status.g_converged

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

    # println(x_abschange(status))
    # println(x_relchange(status))

    # println(x_converged)
    # println(f_converged)
    # println(g_converged)

    return isconverged(status)
end

function meets_stopping_criteria(status::OptimizerStatus, config::Options)
    converged = assess_convergence!(status, config)

    # println(converged && status.i ≥ config.min_iterations )
    # println(status.f_increased && !config.allow_f_increases )
    # println(status.i ≥ config.max_iterations)
    # println(status.rxₐ > config.x_abstol_break)
    # println(status.rxᵣ > config.x_reltol_break)
    # println(status.rfₐ > config.f_abstol_break)
    # println(status.rfᵣ > config.f_reltol_break)
    # println(status.rg  > config.g_restol_break)
    # println(status.x_isnan)
    # println(status.f_isnan)
    # println(status.g_isnan)

    ( converged && status.i ≥ config.min_iterations ) ||
    ( status.f_increased && !config.allow_f_increases ) ||
      status.i ≥ config.max_iterations ||
      status.rxₐ > config.x_abstol_break ||
      status.rxᵣ > config.x_reltol_break ||
      status.rfₐ > config.f_abstol_break ||
      status.rfᵣ > config.f_reltol_break ||
      status.rg  > config.g_restol_break ||
      status.x_isnan ||
      status.f_isnan ||
      status.g_isnan
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
