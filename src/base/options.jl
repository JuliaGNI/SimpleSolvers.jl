"""
Configurable options with defaults (values 0 and NaN indicate unlimited):
```
x_abstol::Real = -Inf,
x_reltol::Real = 2eps(),
f_abstol::Real = 1e-50,
f_reltol::Real = 2eps(),
f_mindec::Real = 1e-4,
g_restol::Real = sqrt(eps()),
x_abstol_break::Real = Inf,
x_reltol_break::Real = Inf,
f_abstol_break::Real = Inf,
f_reltol_break::Real = Inf,
g_restol_break::Real = Inf,
f_calls_limit::Int = 0,
g_calls_limit::Int = 0,
h_calls_limit::Int = 0,
allow_f_increases::Bool = true,
min_iterations::Int = 0,
max_iterations::Int = 1_000,
warn_iterations::Int = max_iterations,
show_trace::Bool = false,
store_trace::Bool = false,
extended_trace::Bool = false,
show_every::Int = 1,
verbosity::Int = 1
```
"""
struct Options{T}
    x_abstol::T
    x_reltol::T
    x_suctol::T
    f_abstol::T
    f_reltol::T
    f_suctol::T
    f_mindec::T
    g_restol::T
    x_abstol_break::T
    x_reltol_break::T
    f_abstol_break::T
    f_reltol_break::T
    g_restol_break::T
    f_calls_limit::Int
    g_calls_limit::Int
    h_calls_limit::Int
    allow_f_increases::Bool
    min_iterations::Int
    max_iterations::Int
    warn_iterations::Int
    show_trace::Bool
    store_trace::Bool
    extended_trace::Bool
    show_every::Int
    verbosity::Int
end

function Options(;
        x_tol = nothing,
        f_tol = nothing,
        g_tol = nothing,
        x_abstol::Real = -Inf,
        x_reltol::Real = 2eps(),
        x_suctol::Real = 2eps(),
        f_abstol::Real = 1e-50,
        f_reltol::Real = 2eps(),
        f_suctol::Real = 2eps(),
        f_mindec::Real = 1e-4,
        g_restol::Real = sqrt(eps()),
        x_abstol_break::Real = Inf,
        x_reltol_break::Real = Inf,
        f_abstol_break::Real = Inf,
        f_reltol_break::Real = Inf,
        g_restol_break::Real = Inf,
        f_calls_limit::Int = 0,
        g_calls_limit::Int = 0,
        h_calls_limit::Int = 0,
        allow_f_increases::Bool = true,
        min_iterations::Int = 0,
        max_iterations::Int = 1_000,
        warn_iterations::Int = max_iterations,
        show_trace::Bool = false,
        store_trace::Bool = false,
        extended_trace::Bool = false,
        show_every::Int = 1,
        verbosity::Int = 1)

    show_every = show_every > 0 ? show_every : 1
    
    if extended_trace
       show_trace = true
    end
    if !(x_tol === nothing)
        x_abstol = x_tol
    end
    if !(g_tol === nothing)
        g_restol = g_tol
    end
    if !(f_tol === nothing)
        f_reltol = f_tol
    end

    Options(promote(x_abstol, x_reltol, x_suctol, f_abstol, f_reltol, f_suctol, f_mindec, g_restol,
                    x_abstol_break, x_reltol_break, f_abstol_break, f_reltol_break, g_restol_break)...,
        f_calls_limit, g_calls_limit, h_calls_limit, allow_f_increases, min_iterations, max_iterations, warn_iterations,
        show_trace, store_trace, extended_trace, show_every, verbosity)
end

function Options(T, options)

    floatopts = (
        options.x_abstol,
        options.x_reltol,
        options.x_suctol,
        options.f_abstol,
        options.f_reltol,
        options.f_suctol,
        options.f_mindec,
        options.g_restol,
        options.x_abstol_break,
        options.x_reltol_break,
        options.f_abstol_break,
        options.f_reltol_break,
        options.g_restol_break,
    )

    nonfloats = (
        options.f_calls_limit,
        options.g_calls_limit,
        options.h_calls_limit,
        options.allow_f_increases,
        options.min_iterations,
        options.max_iterations,
        options.warn_iterations,
        options.show_trace,
        options.store_trace,
        options.extended_trace,
        options.show_every,
        options.verbosity,
    )

    Options(map(x -> convert(T, x), floatopts)..., nonfloats...)
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
g_restol(o::Options) = o.g_restol

verbosity(o::Options) = o.verbosity
