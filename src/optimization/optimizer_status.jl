
mutable struct OptimizerStatus{XT,YT,VT}
    i::Int  # iteration number

    rxₐ::XT  # residual (absolute)
    rxᵣ::XT  # residual (relative)
    rfₐ::YT  # residual (absolute)
    rfᵣ::YT  # residual (relative)
    rgₐ::XT  # residual (absolute)
    rgᵣ::XT  # residual (relative)

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

OptimizerStatus{XT,YT,VT}(n) where {XT,YT,VT} = OptimizerStatus{XT,YT,VT}(
    0, XT(NaN), XT(NaN), YT(NaN), YT(NaN), XT(NaN), XT(NaN),
    zeros(XT,n), zeros(XT,n), zeros(XT,n),
    YT(NaN), YT(NaN),
    zeros(XT,n), zeros(XT,n), zeros(XT,n),
    false, false, false, false)

OptimizerStatus{T}(n) where {T} = OptimizerStatus{T,T,Vector{T}}(n)

solution(status::OptimizerStatus) = status.x

function clear!(status::OptimizerStatus{XT,YT}) where {XT,YT}
    status.i = 0

    status.rxₐ = XT(NaN)
    status.rxᵣ = XT(NaN)
    status.rfₐ = YT(NaN)
    status.rfᵣ = YT(NaN)
    status.rgₐ = XT(NaN)
    status.rgᵣ = XT(NaN)

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

Base.show(io::IO, status::OptimizerStatus) = print(io,
                        (@sprintf "    i=%4i" status.i),  ",   ",
                        (@sprintf "rxₐ=%14.8e" status.rxₐ), ",   ",
                        (@sprintf "rxᵣ=%14.8e" status.rxᵣ), ",   ",
                        (@sprintf "ryₐ=%14.8e" status.rfₐ), ",   ",
                        (@sprintf "ryᵣ=%14.8e" status.rfᵣ), ",   ",
                        (@sprintf "rgₐ=%14.8e" status.rgₐ), ",   ",
                        (@sprintf "rgᵣ=%14.8e" status.rgᵣ), ",   ",
                    )

function print_status(status::OptimizerStatus, config::Options)
    if (config.verbosity ≥ 1 && !(assess_convergence(status, config) && status.i ≤ config.max_iterations)) ||
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

    status.f_increased = status.f > status.f̄

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
    status.rfₐ = norm(status.f̄ - status.f)
    status.rfᵣ = status.rfₐ / norm(status.f̄)

    status.δ .= status.x̄ .- status.x
    status.rxₐ = norm(status.δ)
    status.rxᵣ = status.rxₐ / norm(status.x̄)

    status.γ  .= status.ḡ .- status.g
    status.rgₐ = norm(status.γ)
    status.rgᵣ = status.rgₐ / norm(status.ḡ)
end

function residual!(status::OptimizerStatus, x, f, g)
    status.x .= x
    status.f  = f
    status.g .= g
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
