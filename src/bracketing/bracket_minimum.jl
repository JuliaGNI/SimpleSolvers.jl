"Default constant."
const DEFAULT_BRACKETING_s = 1E-2
"Default constant."
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
function bracket_minimum(f, x=0.0; s=DEFAULT_BRACKETING_s, k=DEFAULT_BRACKETING_k, nmax=DEFAULT_BRACKETING_nmax)
    a, ya = x, f(x)
    b, yb = a + s, f(a + s)
    if yb > ya
        a, b = b, a
        ya, yb = yb, ya
        s = -s
    end
    for _ in 1:nmax
        c, yc = b + s, f(b + s)
        # println("a=$a, b=$b, c=$c, f(a)=$ya, f(b)=$yb, f(c)=$yc")
        if yc > yb
            return a < c ? (a,c) : (c,a)
        end
        a, ya, b, yb = b, yb, c, yc
        s *= k
    end
    error("Unable to bracket f starting at x = $x.")
end
