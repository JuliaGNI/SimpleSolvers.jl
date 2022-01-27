"""
Configurable options with defaults (values 0 and NaN indicate unlimited):
```
x_abstol::Real = 1e-50,
x_reltol::Real = 2eps(),
f_abstol::Real = 1e-50,
f_reltol::Real = 2eps(),
g_abstol::Real = sqrt(eps()),
g_reltol::Real = sqrt(eps()),
x_abstol_break::Real = Inf,
x_reltol_break::Real = Inf,
f_abstol_break::Real = Inf,
f_reltol_break::Real = Inf,
g_abstol_break::Real = Inf,
g_reltol_break::Real = Inf,
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
    f_abstol::T
    f_reltol::T
    g_abstol::T
    g_reltol::T
    x_abstol_break::T
    x_reltol_break::T
    f_abstol_break::T
    f_reltol_break::T
    g_abstol_break::T
    g_reltol_break::T
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
        x_abstol::Real = 1e-50,
        x_reltol::Real = 2eps(),
        f_abstol::Real = 1e-50,
        f_reltol::Real = 2eps(),
        g_abstol::Real = sqrt(eps()),
        g_reltol::Real = sqrt(eps()),
        x_abstol_break::Real = Inf,
        x_reltol_break::Real = Inf,
        f_abstol_break::Real = Inf,
        f_reltol_break::Real = Inf,
        g_abstol_break::Real = Inf,
        g_reltol_break::Real = Inf,
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
        g_abstol = g_tol
    end
    if !(f_tol === nothing)
        f_reltol = f_tol
    end

    Options(promote(x_abstol, x_reltol, f_abstol, f_reltol, g_abstol, g_reltol,
                    x_abstol_break, x_reltol_break, f_abstol_break, f_reltol_break, g_abstol_break, g_reltol_break)...,
        f_calls_limit, g_calls_limit, h_calls_limit, allow_f_increases, min_iterations, max_iterations, warn_iterations,
        show_trace, store_trace, extended_trace, show_every, verbosity)
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
f_abstol(o::Options) = o.f_abstol
f_reltol(o::Options) = o.f_reltol
g_abstol(o::Options) = o.g_abstol
g_reltol(o::Options) = o.g_reltol
