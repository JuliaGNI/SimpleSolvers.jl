
import NLsolve: OnceDifferentiable, NewtonCache, newton_
import LineSearches


struct NLsolveNewton{T, FT, DT, CT, ST, LT} <: AbstractNewtonSolver{T}
    x::Vector{T}
    f::Vector{T}
    J::Matrix{T}

    cache::CT

    F!::FT
    DF::DT

    line_search::ST
    linear_solver::LT

    params::NonlinearSolverParameters{T}
    status::NonlinearSolverStatus{T}

    function NLsolveNewton(x::AbstractVector{T}, f::AbstractVector{T}, J::AbstractMatrix{T},
                    F!::FT, DF::DT, cache::CT, line_search::ST, linear_solver::LT) where {T,FT,DT,CT,ST,LT}

        nls_params = NonlinearSolverParameters(T)
        nls_status = NonlinearSolverStatus{T}(length(x))

        new{T,FT,DT,CT,ST,LT}(x, f, J, cache, F!, DF, line_search, linear_solver, nls_params, nls_status)
    end
end


function NLsolveNewton(x::AbstractVector{T}, f::AbstractVector{T}, F!::Function; J!::Union{Function,Nothing}=nothing) where {T}
    linear_solver = getLinearSolver(x)

    df = linear_solver.A

    if J! === nothing
        if get_config(:jacobian_autodiff)
            DF = OnceDifferentiable(F!, x, f, df; autodiff=:forward, inplace=true)
        else
            DF = OnceDifferentiable(F!, x, f, df; autodiff=:finite, inplace=true)
        end
    else
        DF = OnceDifferentiable(F!, J!, x, f, df; inplace=true)
    end

    NLsolveNewton(x, f, df, F!, DF, NewtonCache(DF), LineSearches.Static(), linear_solver)
end


function linsolve!(s::NLsolveNewton, x, A, b)
    copyto!(s.linear_solver.A, A)
    copyto!(s.linear_solver.b, b)
    factorize!(s.linear_solver)
    solve!(s.linear_solver)
    copyto!(x, s.linear_solver.b)
end

function solve!(s::NLsolveNewton; n::Int=0)
    local nmax::Int = n > 0 ? nmax = n : s.params.nmax

    res=newton_(s.DF, s.x, s.params.stol, s.params.atol, nmax, false, false, false,
                s.line_search, (x, A, b) -> linsolve!(s, x, A, b), s.cache)

    s.x .= res.zero
    s.status.i  = res.iterations
    s.status.rₐ = res.residual_norm

    nothing
end
