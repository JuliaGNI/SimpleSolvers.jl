module SimpleSolversRecursiveFactorizationExt

using LinearAlgebra: BlasInt
using RecursiveFactorization: RecursiveFactorization

using SimpleSolvers: RecursiveLU, PivotedLUCache, _pivoted_lu_cache
import SimpleSolvers: LinearSolverCache, _getrf!

function LinearSolverCache(::RecursiveLU, A::AbstractMatrix{T}) where {T}
    T <: Union{Float32,Float64} || throw(ArgumentError(
        "RecursiveLU is restricted to Float32 and Float64 — RecursiveFactorization has no " *
        "complex support — but got $(T); use LapackLU() for ComplexF32/ComplexF64 or LU() " *
        "for anything else"))
    _pivoted_lu_cache(A)
end

# `Val(true)` is partial pivoting, which is what the shared `getrs` solve and
# `singular_index` assume. `Val(false)` is the threading flag: see the `RecursiveLU`
# docstring for why the threaded path is not offered.
#
# `lu!` returns a `LinearAlgebra.LU` wrapping the arrays we already own, so only its `info` is
# kept. That wrapper is the one allocation RecursiveFactorization makes here and Julia elides
# it, leaving `factorize!` allocation-free — which the tests assert.
function _getrf!(::RecursiveLU, c::PivotedLUCache)
    F = RecursiveFactorization.lu!(c.A, c.ipiv, Val(true), Val(false); check=false)
    c.info = BlasInt(F.info)
    c
end

end # module
