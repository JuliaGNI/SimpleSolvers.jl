
abstract type Optimizer{T} <: NonlinearSolver{T} end



mutable struct OptimizerStatus{XT,YT,VT}
    i::Int  # iteration number

    rxₐ::XT  # residual (absolute)
    rxᵣ::XT  # residual (relative)
    ryₐ::YT  # residual (absolute)
    ryᵣ::YT  # residual (relative)
    rgₐ::XT  # residual (absolute)
    rgᵣ::XT  # residual (relative)

    x̄::VT    # previous solution
    x::VT    # current solution
    δ::VT

    ȳ::YT    # previous function
    y::YT    # current function

    ḡ::VT    # previous gradient
    g::VT    # current gradient
    γ::VT

end

OptimizerStatus{XT,YT,VT}(n) where {XT,YT,VT} = OptimizerStatus{XT,YT,VT}(
            0, 0, 0, 0, 0, 0, 0,
            zeros(XT,n), zeros(XT,n), zeros(XT,n),
            0, 0, 
            zeros(XT,n), zeros(XT,n), zeros(XT,n))

OptimizerStatus{T}(n) where {T} = OptimizerStatus{T,T,Vector{T}}(n)
                

Base.show(io::IO, status::OptimizerStatus) = print(io,
                        (@sprintf "    i=%4i" status.i),  ",   ",
                        (@sprintf "rxₐ=%14.8e" status.rxₐ), ",   ",
                        (@sprintf "rxᵣ=%14.8e" status.rxᵣ), ",   ",
                        (@sprintf "ryₐ=%14.8e" status.ryₐ), ",   ",
                        (@sprintf "ryᵣ=%14.8e" status.ryᵣ), ",   ",
                        (@sprintf "rgₐ=%14.8e" status.rgₐ), ",   ",
                        (@sprintf "rgᵣ=%14.8e" status.rgᵣ), ",   ",
                    )

function print_solver_status(status::OptimizerStatus, config::Options)
    if (config.verbosity ≥ 1 && !(check_solver_converged(status, config) && status.i ≤ config.max_iterations)) ||
        config.verbosity > 1
        println(status)
    end
end

function check_solver_converged(status::OptimizerStatus, config::Options)
    return (status.rxₐ ≤ config.x_abstol  ||
            status.rxᵣ ≤ config.x_reltol  ||
            status.ryₐ ≤ config.f_abstol  ||
            status.ryᵣ ≤ config.f_reltol  ||
            status.rgₐ ≤ config.g_abstol  ||
            status.rgᵣ ≤ config.g_reltol) &&
           status.i ≥ config.min_iterations
end

function check_solver_status(status::OptimizerStatus, params::NonlinearSolverParameters)
    if any(x -> isnan(x), status.x) || isnan(status.y)
        throw(NonlinearSolverException("Detected NaN"))
    end

    if status.ryₐ > params.atol_break
        throw(NonlinearSolverException("Absolute error ($(status.ryₐ)) larger than allowed ($(params.atol_break))"))
    end

    if status.ryᵣ > params.rtol_break
        throw(NonlinearSolverException("Relative error ($(status.ryᵣ)) larger than allowed ($(params.rtol_break))"))
    end
end

function get_solver_status!(status::OptimizerStatus, params::NonlinearSolverParameters, status_dict::Dict)
    status_dict[:niter] = status.i
    status_dict[:xatol] = status.rxₐ
    status_dict[:xrtol] = status.rxᵣ
    status_dict[:yatol] = status.ryₐ
    status_dict[:yrtol] = status.ryᵣ
    status_dict[:gatol] = status.rgₐ
    status_dict[:grtol] = status.rgᵣ
    status_dict[:converged] = check_solver_converged(status, params)
    return status_dict
end

get_solver_status!(solver::OptimizerStatus{T}, status_dict::Dict) where {T} =
            get_solver_status!(status(solver), params(solver), status_dict)

get_solver_status(solver::OptimizerStatus{T}) where {T} = get_solver_status!(solver,
            Dict(:niter => 0,
                 :xatol => zero(T),
                 :xrtol => zero(T),
                 :yatol => zero(T),
                 :yrtol => zero(T),
                 :gatol => zero(T),
                 :grtol => zero(T),
                 :converged => false)
            )


function warn_iteration_number(status, config)
    if config.warn_iterations > 0 && status.i ≥ config.warn_iterations
        println("WARNING: Optimizer took ", status.i, " iterations.")
    end
end


function residual!(status::OptimizerStatus)
    status.ryₐ = norm(status.ȳ - status.y)
    status.ryᵣ = status.ryₐ / norm(status.ȳ)

    status.δ .= status.x̄ .- status.x
    status.rxₐ = norm(status.δ)
    status.rxᵣ = status.rxₐ / norm(status.x̄)

    status.γ  .= status.ḡ .- status.g
    status.rgₐ = norm(status.γ)
    status.rgᵣ = status.rgₐ / norm(status.ḡ)
end

function residual!(status::OptimizerStatus, x, y, g)
    status.x .= x
    status.y  = y
    status.g .= g
    residual!(status)
end

function residual_initial!(status::OptimizerStatus)
    status.rxₐ = Inf
    status.rxᵣ = Inf
    status.ryₐ = Inf
    status.ryᵣ = Inf
    status.rgₐ = Inf
    status.rgᵣ = Inf
end

function update!(status::OptimizerStatus)
    status.x̄ .= status.x
    status.ȳ  = status.y
    status.ḡ .= status.g
end

function initialize!(status::OptimizerStatus, x, y, g)
    status.x̄ .= status.x .= x
    status.ȳ  = status.y  = y
    status.ḡ .= status.g .= g
    residual_initial!(status)
end
