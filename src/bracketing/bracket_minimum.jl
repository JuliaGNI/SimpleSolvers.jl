"""
    const DEFAULT_BRACKETING_s

Gives the default width of the interval (the bracket). See [`bracket_minimum`](@ref).
"""
const DEFAULT_BRACKETING_s = 1E-2

"""
    const DEFAULT_BRACKETING_k

Gives the default ratio by which the bracket is increased if bracketing was not successful. See [`bracket_minimum`](@ref).
"""
const DEFAULT_BRACKETING_k = 2.0

"Default constant"
const DEFAULT_BRACKETING_nmax=100

"""
    bracket_minimum(f, x)

# Keyword arguments
- `s::`[`DEFAULT_BRACKETING_s`](@ref)
- `k::`[`DEFAULT_BRACKETING_k`](@ref)
- `nmax::`[`DEFAULT_BRACKETING_nmax`](@ref)
"""
function bracket_minimum(f::Callable, x::T=0.0; s::T=T(DEFAULT_BRACKETING_s), k::T=T(DEFAULT_BRACKETING_k), nmax::Integer=DEFAULT_BRACKETING_nmax)::Tuple{T, T} where {T <: Number}
    a = x
    ya = f(a)

    b = a + s
    yb = f(b)

    # flip a & b if necessary
    if yb > ya
        a, b = b, a
        ya, yb = yb, ya
        s = -s
    end

    for _ in 1:nmax
        c = b + s
        yc = f(c)
        if yc > yb
            interval = a < c ? (a, c) : (c, a)
            return interval
        end
        a = b
        ya = yb
        b = c
        yb = yc
        s *= k
    end
    error("Unable to bracket f starting at x = $x.")
end
